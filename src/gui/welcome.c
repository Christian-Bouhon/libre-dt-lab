/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <pango/pangocairo.h>

#include "bauhaus/bauhaus.h"
#include "welcome.h"
#include "common/darktable.h"
#include "common/usermanual_url.h"
#include "control/conf.h"
#include "dtgtk/button.h"
#include "gui/gtk.h"
#ifdef GDK_WINDOWING_QUARTZ
#include "osx/osx.h"
#endif

#define LOGO_SIZE 80

// ── internal types ────────────────────────────────────────────────────────────

// Action that fires when a checkbox is toggled.
typedef enum
{
  DT_WELCOME_ACTION_NONE,
  DT_WELCOME_ACTION_SET_BOOL,
} _dt_welcome_action_type_t;

typedef struct
{
  _dt_welcome_action_type_t type;
  const char *key;
} _dt_welcome_action_t;

typedef enum
{
  DT_QUESTION_CHECKBOX,
  DT_QUESTION_DIRCHOOSER,
  DT_QUESTION_PARAGRAPH,
  DT_QUESTION_COMBOBOX,
  DT_QUESTION_THEME,
} _dt_question_type_t;

typedef struct
{
  GPtrArray *questions; // array of _dt_question_t *
} _dt_page_t;

typedef struct
{
  _dt_question_type_t qtype;
  char *label;       // widget label text
  char *description; // optional, HTML subset → rendered as Pango markup
  gboolean centered; // for DT_QUESTION_PARAGRAPH: centre-align text
  // CHECKBOX only:
  gboolean default_active;
  _dt_welcome_action_t action;
  // DIRCHOOSER and COMBOBOX:
  char *conf_key;
  // COMBOBOX only:
  char **options;        // array of option ids (written to conf)
  char **option_labels;  // parallel array of display labels (NULL entry
                         // when an option has no "id|label" form; in that
                         // case the id itself is used as the display text).
                         // labels are looked up via g_dpgettext2 at display
                         // time so translators can localize them
  int n_options;
  // THEME only (widgets are owned by GTK, not freed here):
  GtkWidget *combo_base;
  GtkWidget *combo_accent;
  GtkWidget *combo_density;
  GtkWidget *preview;
} _dt_question_t;

// ── theme picker data (fixed lists) ───────────────────────────────────────────

static const char *THEME_BASES[]     = { "ardoise", "anthracite" };
static const int   THEME_N_BASES     = G_N_ELEMENTS(THEME_BASES);
static const char *THEME_ACCENTS[]   = { "neutral", "cyan", "amber", "red", "teal" };
static const int   THEME_N_ACCENTS   = G_N_ELEMENTS(THEME_ACCENTS);

// translatable labels (ids above stay untranslated for the conf value)
static const char *THEME_BASE_LABELS[]    = { N_("ardoise"), N_("anthracite") };
static const char *THEME_ACCENT_LABELS[]  = { N_("neutral"), N_("cyan"), N_("amber"),
                                              N_("red"), N_("teal") };
static const char *THEME_DENSITY_LABELS[] = { N_("normal"), N_("compact (laptop)") };
static const int   THEME_N_DENSITIES      = G_N_ELEMENTS(THEME_DENSITY_LABELS);

// colours used by the mini preview, indexed by base then accent
typedef struct
{
  const char *bg;
  const char *fg;
  const char *accent[5];
} _theme_preview_t;

static const _theme_preview_t THEME_PREVIEW[] = {
  // ardoise (medium neutral grey)
  { "#4e4e4e", "#f3f3f3", { "#c6c6c6", "#7db8d6", "#d9a441", "#e9906f", "#69b3a2" } },
  // anthracite (dark charcoal)
  { "#1c1f24", "#e8ebf0", { "#c6c6c6", "#4fb3d9", "#e6a23c", "#f65b3c", "#35b9a0" } },
};

struct _dt_welcome_screen_t
{
  GPtrArray *pages; // array of _dt_page_t *
};

// ── helpers ───────────────────────────────────────────────────────────────────

static void _free_question(gpointer p)
{
  _dt_question_t *q = p;
  g_free(q->label);
  g_free(q->description);
  if(q->qtype == DT_QUESTION_DIRCHOOSER || q->qtype == DT_QUESTION_COMBOBOX)
  {
    g_free(q->conf_key);
    if(q->qtype == DT_QUESTION_COMBOBOX)
    {
      for(int i = 0; i < q->n_options; i++)
      {
        g_free(q->options[i]);
        if(q->option_labels) g_free(q->option_labels[i]);
      }
      g_free(q->options);
      g_free(q->option_labels);
    }
  }
  else if(q->qtype == DT_QUESTION_CHECKBOX && q->action.type != DT_WELCOME_ACTION_NONE)
    g_free((char *)q->action.key);
  g_free(q);
}

static void _free_page(gpointer p)
{
  _dt_page_t *pg = p;
  g_ptr_array_free(pg->questions, TRUE);
  g_free(pg);
}

// ── logo ──────────────────────────────────────────────────────────────────────

static GtkWidget *_get_logo(void)
{
  const dt_logo_season_t season = dt_util_get_logo_season();
  gchar *image_file =
      season == DT_LOGO_SEASON_NONE
          ? g_strdup_printf("%s/pixmaps/idbutton.svg", darktable.datadir)
          : g_strdup_printf("%s/pixmaps/idbutton-%d.svg", darktable.datadir, season);

  GdkPixbuf *pb = gdk_pixbuf_new_from_file_at_size(image_file, LOGO_SIZE, -1, NULL);
  g_free(image_file);

  GtkWidget *logo;
  if(pb)
  {
    logo = gtk_image_new_from_pixbuf(pb);
    g_object_unref(pb);
  }
  else
    logo = gtk_label_new("libre-dt-lab");

  gtk_widget_set_name(logo, "welcome-logo");
  gtk_widget_set_halign(logo, GTK_ALIGN_CENTER);
  return logo;
}

// ── checkbox callback ─────────────────────────────────────────────────────────

static void _on_answer_toggled(GtkToggleButton *btn, gpointer data)
{
  const _dt_welcome_action_t *action = data;
  if(action->type == DT_WELCOME_ACTION_SET_BOOL)
    dt_conf_set_bool(action->key, gtk_toggle_button_get_active(btn));
}

// ── dirchooser callbacks ──────────────────────────────────────────────────────

static void _on_dir_entry_changed(GtkEntry *entry, gpointer data)
{
  dt_conf_set_string((const char *)data, gtk_entry_get_text(entry));
}

static void _on_browse_dir_clicked(GtkWidget *btn, gpointer data)
{
  GtkEntry *entry = GTK_ENTRY(data);
  GtkWidget *topwindow = gtk_widget_get_toplevel(btn);
  if(!GTK_IS_WINDOW(topwindow))
    topwindow = dt_ui_main_window(darktable.gui->ui);

  GtkFileChooserNative *fc = gtk_file_chooser_native_new(
      _("select directory"), GTK_WINDOW(topwindow),
      GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, _("_open"), _("_cancel"));

  // Strip pattern variables (e.g. "$(PICTURES_FOLDER)/...") before
  // passing the path to the chooser.
  gchar *old = g_strdup(gtk_entry_get_text(entry));
  char *dollar = g_strstr_len(old, -1, "$");
  if(dollar)
    *dollar = '\0';
  if(*old)
    gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(fc), old);
  g_free(old);

  if(gtk_native_dialog_run(GTK_NATIVE_DIALOG(fc)) == GTK_RESPONSE_ACCEPT)
  {
    gchar *dir = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(fc));
    // Escape backslashes (needed on Windows; harmless elsewhere).
    gchar *escaped = dt_util_str_replace(dir, "\\", "\\\\");
    gtk_entry_set_text(entry, escaped); // "changed" signal writes to conf
    g_free(dir);
    g_free(escaped);
  }
  g_object_unref(fc);
}

// ── combobox callback ─────────────────────────────────────────────────────────

static void _on_combo_changed(GtkComboBox *combo, gpointer data)
{
  const _dt_question_t *q = data;
  const int idx = gtk_combo_box_get_active(combo);
  if(idx >= 0 && idx < q->n_options)
  {
    dt_conf_set_string(q->conf_key, q->options[idx]);
    // theme applies live so the user can preview grey/dark/darker
    // and judge which surround they prefer before moving on
    if(!g_strcmp0(q->conf_key, "ui_last/theme"))
    {
      dt_gui_load_theme(q->options[idx]);
      dt_gui_apply_theme();
    }
  }
}

// ── theme picker callbacks ────────────────────────────────────────────────────

// Parse a "{base}-modern-{accent}[-compact]" theme name. Anything that does
// not follow the convention leaves the defaults (ardoise/neutral/normal).
static void _theme_parse(const char *name, int *base, int *accent, int *density)
{
  *base = 0;    // ardoise
  *accent = 0;  // neutral
  *density = 0; // normal

  if(!name) return;

  const char *modern = strstr(name, "-modern-");
  if(!modern) return;

  if(g_str_has_prefix(name, "ardoise-"))
    *base = 0;
  else if(g_str_has_prefix(name, "anthracite-"))
    *base = 1;
  else
    return; // unknown family: keep defaults and don't touch conf

  *density = g_str_has_suffix(name, "-compact") ? 1 : 0;

  const char *a = modern + strlen("-modern-");
  gsize len = strlen(a);
  if(g_str_has_suffix(a, "-compact"))
    len -= strlen("-compact");

  for(int i = 0; i < THEME_N_ACCENTS; i++)
  {
    if(len == strlen(THEME_ACCENTS[i])
       && !g_ascii_strncasecmp(a, THEME_ACCENTS[i], len))
    {
      *accent = i;
      break;
    }
  }
}

static void _theme_compose(const int base, const int accent, const int density,
                           char *out, const gsize out_size)
{
  g_snprintf(out, out_size, "%s-modern-%s%s",
             THEME_BASES[base], THEME_ACCENTS[accent],
             density ? "-compact" : "");
}

static void _on_theme_changed(GtkComboBox *combo, gpointer data)
{
  _dt_question_t *q = data;
  (void)combo;

  if(!q->combo_base || !q->combo_accent || !q->combo_density) return;

  const int base = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(q->combo_base)),
                         0, THEME_N_BASES - 1);
  const int accent = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(q->combo_accent)),
                           0, THEME_N_ACCENTS - 1);
  const int density = CLAMP(gtk_combo_box_get_active(GTK_COMBO_BOX(q->combo_density)),
                            0, THEME_N_DENSITIES - 1);

  char name[128];
  _theme_compose(base, accent, density, name, sizeof(name));

  // apply live so the user can judge the surround before moving on
  dt_conf_set_string("ui_last/theme", name);
  dt_gui_load_theme(name);
  dt_gui_apply_theme();
  dt_bauhaus_load_theme();

  if(q->preview) gtk_widget_queue_draw(q->preview);
}

static gboolean _on_theme_preview_draw(GtkWidget *widget, cairo_t *cr, gpointer data)
{
  _dt_question_t *q = data;
  if(!q->combo_base || !q->combo_accent) return FALSE;

  int base = gtk_combo_box_get_active(GTK_COMBO_BOX(q->combo_base));
  int accent = gtk_combo_box_get_active(GTK_COMBO_BOX(q->combo_accent));
  if(base < 0 || base >= (int)G_N_ELEMENTS(THEME_PREVIEW)) base = 0;
  if(accent < 0 || accent >= THEME_N_ACCENTS) accent = 0;

  GdkRGBA bg, fg, acc;
  gdk_rgba_parse(&bg, THEME_PREVIEW[base].bg);
  gdk_rgba_parse(&fg, THEME_PREVIEW[base].fg);
  gdk_rgba_parse(&acc, THEME_PREVIEW[base].accent[accent]);

  const double width = gtk_widget_get_allocated_width(widget);
  const double height = gtk_widget_get_allocated_height(widget);
  const double r = 6.0;

  // background
  gdk_cairo_set_source_rgba(cr, &bg);
  cairo_new_sub_path(cr);
  cairo_arc(cr, width - r, r, r, -G_PI_2, 0);
  cairo_arc(cr, width - r, height - r, r, 0, G_PI_2);
  cairo_arc(cr, r, height - r, r, G_PI_2, G_PI);
  cairo_arc(cr, r, r, r, G_PI, 3 * G_PI_2);
  cairo_close_path(cr);
  cairo_fill(cr);

  // accent capsule (mock slider / active control)
  const double bar_x = 10.0;
  const double bar_y = 9.0;
  const double bar_h = 8.0;
  const double bar_w = MAX(bar_h, width - 20.0);
  gdk_cairo_set_source_rgba(cr, &acc);
  cairo_new_sub_path(cr);
  cairo_arc(cr, bar_x + bar_h / 2.0, bar_y + bar_h / 2.0, bar_h / 2.0, -G_PI_2, G_PI_2);
  cairo_arc(cr, bar_x + bar_w - bar_h / 2.0, bar_y + bar_h / 2.0, bar_h / 2.0,
            G_PI_2, 3 * G_PI_2);
  cairo_close_path(cr);
  cairo_fill(cr);

  // sample text in the theme foreground colour
  PangoLayout *layout = gtk_widget_create_pango_layout(widget, "Aa");
  gdk_cairo_set_source_rgba(cr, &fg);
  cairo_move_to(cr, 10.0, height - 24.0);
  pango_cairo_show_layout(cr, layout);
  g_object_unref(layout);

  return FALSE;
}

// A [bold label][combo] row; returns the row and sets *out_combo.
static GtkWidget *_theme_add_row(GtkWidget *col, const char *label_text,
                                 GtkWidget **out_combo)
{
  GtkWidget *row = dt_gui_hbox();

  GtkWidget *lbl = gtk_label_new(label_text);
  gtk_widget_set_name(lbl, "welcome-answer");
  gtk_widget_set_halign(lbl, GTK_ALIGN_START);
  gtk_widget_set_valign(lbl, GTK_ALIGN_CENTER);
  gtk_widget_set_hexpand(lbl, TRUE);
  {
    char *esc = g_markup_escape_text(label_text, -1);
    char *mu = g_strdup_printf("<b>%s</b>", esc);
    gtk_label_set_markup(GTK_LABEL(lbl), mu);
    g_free(esc);
    g_free(mu);
  }

  GtkWidget *combo = gtk_combo_box_text_new();
  gtk_widget_set_name(combo, "welcome-combo");
  gtk_widget_set_halign(combo, GTK_ALIGN_END);
  gtk_widget_set_valign(combo, GTK_ALIGN_CENTER);

  dt_gui_box_add(GTK_BOX(row), lbl, combo);
  dt_gui_box_add(GTK_BOX(col), row);

  if(out_combo) *out_combo = combo;
  return row;
}

// ── navigation state ──────────────────────────────────────────────────────────

typedef struct
{
  GtkWidget *stack;
  GtkWidget *prev_btn;
  GtkWidget *next_btn;
  GtkWidget *progress_box;
  int n_stack_pages; // user pages + final page
  int current;
} _nav_t;

static void _update_navigation(_nav_t *nav)
{
  // Update stack
  GList *children = gtk_container_get_children(GTK_CONTAINER(nav->stack));
  GtkWidget *target = g_list_nth_data(children, nav->current);
  if(target)
    gtk_stack_set_visible_child(GTK_STACK(nav->stack), target);
  g_list_free(children);

  // Prev / Next: always visible, just grayed out when unavailable.
  const gboolean is_last = (nav->current == nav->n_stack_pages - 1);
  gtk_widget_set_sensitive(nav->prev_btn, nav->current > 0);
  gtk_widget_set_sensitive(nav->next_btn, !is_last);

  // Progress dots
  GList *dots = gtk_container_get_children(GTK_CONTAINER(nav->progress_box));
  int idx = 0;
  for(GList *l = dots; l; l = l->next, idx++)
  {
    GtkWidget *dot = l->data;
    gtk_label_set_text(GTK_LABEL(dot), idx == nav->current ? "●" : "○");
  }
  g_list_free(dots);
}

static void _on_prev(GtkWidget *btn, gpointer data)
{
  (void)btn;
  _nav_t *nav = data;
  if(nav->current > 0)
  {
    nav->current--;
    _update_navigation(nav);
  }
}

static void _on_next(GtkWidget *btn, gpointer data)
{
  (void)btn;
  _nav_t *nav = data;
  if(nav->current < nav->n_stack_pages - 1)
  {
    nav->current++;
    _update_navigation(nav);
  }
}

// ── page widget builder ───────────────────────────────────────────────────────

static GtkWidget *_build_page_widget(_dt_page_t *pg)
{
  // The box IS the page: vertically centered in the stack's allocated height,
  // so that content floats in the middle rather than sticking to the top.
  GtkWidget *box = dt_gui_vbox();
  gtk_widget_set_name(box, "welcome-page");
  gtk_widget_set_valign(box, GTK_ALIGN_CENTER);

  for(guint qi = 0; qi < pg->questions->len; qi++)
  {
    _dt_question_t *q = pg->questions->pdata[qi];

    if(q->qtype == DT_QUESTION_PARAGRAPH)
    {
      GtkWidget *para = gtk_label_new(NULL);
      gtk_label_set_markup(GTK_LABEL(para), q->label);
      gtk_widget_set_name(para, "welcome-paragraph");
      gtk_label_set_line_wrap(GTK_LABEL(para), TRUE);
      if(q->centered)
      {
        gtk_widget_set_halign(para, GTK_ALIGN_CENTER);
        gtk_label_set_xalign(GTK_LABEL(para), 0.5f);
        gtk_label_set_justify(GTK_LABEL(para), GTK_JUSTIFY_CENTER);
      }
      else
      {
        gtk_widget_set_halign(para, GTK_ALIGN_FILL);
        gtk_label_set_xalign(GTK_LABEL(para), 0.0f);
        gtk_label_set_justify(GTK_LABEL(para), GTK_JUSTIFY_LEFT);
      }
      dt_gui_box_add(GTK_BOX(box), para);
    }
    else if(q->qtype == DT_QUESTION_DIRCHOOSER)
    {
      // Row 1: [bold label (left)] [entry + browse button (right)]
      // Row 2: description, left-aligned
      GtkWidget *col = dt_gui_vbox();
      gtk_widget_set_hexpand(col, TRUE);

      GtkWidget *row1 = dt_gui_hbox();

      GtkWidget *lbl = gtk_label_new(NULL);
      {
        char *esc = g_markup_escape_text(q->label, -1);
        char *mu = g_strdup_printf("<b>%s</b>", esc);
        gtk_label_set_markup(GTK_LABEL(lbl), mu);
        g_free(esc);
        g_free(mu);
      }
      gtk_widget_set_name(lbl, "welcome-answer");
      gtk_widget_set_halign(lbl, GTK_ALIGN_START);
      gtk_widget_set_valign(lbl, GTK_ALIGN_CENTER);

      GtkWidget *hbox = dt_gui_hbox();
      GtkWidget *entry = gtk_entry_new();
      gtk_widget_set_name(entry, "welcome-dir-entry");
      gtk_widget_set_hexpand(entry, TRUE);
      const char *cur = dt_conf_get_string_const(q->conf_key);
      gtk_entry_set_text(GTK_ENTRY(entry), cur ? cur : "");
      g_signal_connect(G_OBJECT(entry), "changed",
                       G_CALLBACK(_on_dir_entry_changed), q->conf_key);

      GtkWidget *btn = dtgtk_button_new(dtgtk_cairo_paint_directory, CPF_NONE, NULL);
      gtk_widget_set_tooltip_text(btn, _("select directory"));
      g_signal_connect(G_OBJECT(btn), "clicked",
                       G_CALLBACK(_on_browse_dir_clicked), entry);
      dt_gui_box_add(GTK_BOX(hbox), entry, btn);

      gtk_widget_set_hexpand(hbox, TRUE);
      dt_gui_box_add(GTK_BOX(row1), lbl, hbox);
      dt_gui_box_add(GTK_BOX(col), row1);

      if(q->description && q->description[0])
      {
        GtkWidget *desc = gtk_label_new(NULL);
        gtk_label_set_markup(GTK_LABEL(desc), q->description);
        gtk_widget_set_name(desc, "welcome-question-description");
        gtk_widget_set_halign(desc, GTK_ALIGN_FILL);
        gtk_widget_set_hexpand(desc, TRUE);
        gtk_label_set_xalign(GTK_LABEL(desc), 0.0f);
        gtk_label_set_line_wrap(GTK_LABEL(desc), TRUE);
        dt_gui_box_add(GTK_BOX(col), desc);
      }

      dt_gui_box_add(GTK_BOX(box), col);
    }
    else if(q->qtype == DT_QUESTION_COMBOBOX)
    {
      // Row 1: [bold label (left)] [combo (right)]
      // Row 2: description, left-aligned
      GtkWidget *col = dt_gui_vbox();
      gtk_widget_set_hexpand(col, TRUE);

      GtkWidget *row1 = dt_gui_hbox();

      GtkWidget *lbl = gtk_label_new(NULL);
      {
        char *esc = g_markup_escape_text(q->label, -1);
        char *mu = g_strdup_printf("<b>%s</b>", esc);
        gtk_label_set_markup(GTK_LABEL(lbl), mu);
        g_free(esc);
        g_free(mu);
      }
      gtk_widget_set_name(lbl, "welcome-answer");
      gtk_widget_set_halign(lbl, GTK_ALIGN_START);
      gtk_widget_set_valign(lbl, GTK_ALIGN_CENTER);
      gtk_widget_set_hexpand(lbl, TRUE);

      GtkWidget *combo = gtk_combo_box_text_new();
      gtk_widget_set_name(combo, "welcome-combo");
      gtk_widget_set_halign(combo, GTK_ALIGN_END);
      gtk_widget_set_valign(combo, GTK_ALIGN_CENTER);
      const char *curval = dt_conf_get_string_const(q->conf_key);
      for(int oi = 0; oi < q->n_options; oi++)
      {
        // display label = the portion after '|' (translated via the
        // "preferences" gettext context); fall back to the id when the
        // entry has no label. conf value is always the id side
        const char *display
          = (q->option_labels && q->option_labels[oi])
              ? q->option_labels[oi]
              : q->options[oi];
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(combo),
                                       g_dpgettext2(NULL, "preferences", display));
        if(g_strcmp0(curval, q->options[oi]) == 0)
          gtk_combo_box_set_active(GTK_COMBO_BOX(combo), oi);
      }
      g_signal_connect(G_OBJECT(combo), "changed", G_CALLBACK(_on_combo_changed), q);

      dt_gui_box_add(GTK_BOX(row1), lbl, combo);
      dt_gui_box_add(GTK_BOX(col), row1);

      if(q->description && q->description[0])
      {
        GtkWidget *desc = gtk_label_new(NULL);
        gtk_label_set_markup(GTK_LABEL(desc), q->description);
        gtk_widget_set_name(desc, "welcome-question-description");
        gtk_widget_set_halign(desc, GTK_ALIGN_FILL);
        gtk_widget_set_hexpand(desc, TRUE);
        gtk_label_set_xalign(GTK_LABEL(desc), 0.0f);
        gtk_label_set_line_wrap(GTK_LABEL(desc), TRUE);
        dt_gui_box_add(GTK_BOX(col), desc);
      }

      dt_gui_box_add(GTK_BOX(box), col);
    }
    else if(q->qtype == DT_QUESTION_THEME)
    {
      GtkWidget *col = dt_gui_vbox();
      gtk_widget_set_hexpand(col, TRUE);

      // panel title
      GtkWidget *title = gtk_label_new(NULL);
      {
        char *esc = g_markup_escape_text(q->label, -1);
        char *mu = g_strdup_printf("<b>%s</b>", esc);
        gtk_label_set_markup(GTK_LABEL(title), mu);
        g_free(esc);
        g_free(mu);
      }
      gtk_widget_set_name(title, "welcome-answer");
      gtk_widget_set_halign(title, GTK_ALIGN_START);
      gtk_widget_set_valign(title, GTK_ALIGN_CENTER);
      dt_gui_box_add(GTK_BOX(col), title);

      if(q->description && q->description[0])
      {
        GtkWidget *desc = gtk_label_new(NULL);
        gtk_label_set_markup(GTK_LABEL(desc), q->description);
        gtk_widget_set_name(desc, "welcome-question-description");
        gtk_widget_set_halign(desc, GTK_ALIGN_FILL);
        gtk_widget_set_hexpand(desc, TRUE);
        gtk_label_set_xalign(GTK_LABEL(desc), 0.0f);
        gtk_label_set_line_wrap(GTK_LABEL(desc), TRUE);
        dt_gui_box_add(GTK_BOX(col), desc);
      }

      // mini preview of the selected theme
      q->preview = gtk_drawing_area_new();
      gtk_widget_set_name(q->preview, "welcome-theme-preview");
      gtk_widget_set_size_request(q->preview, 150, 40);
      gtk_widget_set_valign(q->preview, GTK_ALIGN_CENTER);
      g_signal_connect(G_OBJECT(q->preview), "draw",
                       G_CALLBACK(_on_theme_preview_draw), q);

      // base row
      _theme_add_row(col, _("base"), &q->combo_base);
      for(int i = 0; i < THEME_N_BASES; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(q->combo_base),
                                       _(THEME_BASE_LABELS[i]));

      // accent row (+ preview)
      GtkWidget *accent_row = _theme_add_row(col, _("accent"), &q->combo_accent);
      for(int i = 0; i < THEME_N_ACCENTS; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(q->combo_accent),
                                       _(THEME_ACCENT_LABELS[i]));
      dt_gui_box_add(GTK_BOX(accent_row), q->preview);

      // display density row
      _theme_add_row(col, _("display"), &q->combo_density);
      for(int i = 0; i < THEME_N_DENSITIES; i++)
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(q->combo_density),
                                       _(THEME_DENSITY_LABELS[i]));

      // initial selection: done before connecting so that merely displaying
      // the page does not write anything to the config
      int base = 0, accent = 0, density = 0;
      _theme_parse(dt_conf_get_string_const("ui_last/theme"), &base, &accent, &density);
      gtk_combo_box_set_active(GTK_COMBO_BOX(q->combo_base), base);
      gtk_combo_box_set_active(GTK_COMBO_BOX(q->combo_accent), accent);
      gtk_combo_box_set_active(GTK_COMBO_BOX(q->combo_density), density);

      g_signal_connect(G_OBJECT(q->combo_base), "changed",
                       G_CALLBACK(_on_theme_changed), q);
      g_signal_connect(G_OBJECT(q->combo_accent), "changed",
                       G_CALLBACK(_on_theme_changed), q);
      g_signal_connect(G_OBJECT(q->combo_density), "changed",
                       G_CALLBACK(_on_theme_changed), q);

      dt_gui_box_add(GTK_BOX(box), col);
    }
    else
    {
      // CHECKBOX: row 1 = [bold label (left)] [checkbox (right)]
      //           row 2 = description, left-aligned
      GtkWidget *col = dt_gui_vbox();
      gtk_widget_set_hexpand(col, TRUE);

      GtkWidget *row1 = dt_gui_hbox();

      GtkWidget *lbl = gtk_label_new(NULL);
      {
        char *esc = g_markup_escape_text(q->label, -1);
        char *mu = g_strdup_printf("<b>%s</b>", esc);
        gtk_label_set_markup(GTK_LABEL(lbl), mu);
        g_free(esc);
        g_free(mu);
      }
      gtk_widget_set_name(lbl, "welcome-answer");
      gtk_widget_set_halign(lbl, GTK_ALIGN_START);
      gtk_widget_set_valign(lbl, GTK_ALIGN_CENTER);
      gtk_widget_set_hexpand(lbl, TRUE);

      GtkWidget *cb = gtk_check_button_new();
      gtk_widget_set_name(cb, "welcome-answer");
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(cb), q->default_active);
      gtk_widget_set_valign(cb, GTK_ALIGN_CENTER);
      if(q->action.type != DT_WELCOME_ACTION_NONE)
        g_signal_connect(G_OBJECT(cb), "toggled",
                         G_CALLBACK(_on_answer_toggled), &q->action);

      dt_gui_box_add(GTK_BOX(row1), lbl, cb);
      dt_gui_box_add(GTK_BOX(col), row1);

      if(q->description && q->description[0])
      {
        GtkWidget *desc = gtk_label_new(NULL);
        gtk_label_set_markup(GTK_LABEL(desc), q->description);
        gtk_widget_set_name(desc, "welcome-question-description");
        gtk_widget_set_halign(desc, GTK_ALIGN_FILL);
        gtk_widget_set_hexpand(desc, TRUE);
        gtk_label_set_xalign(GTK_LABEL(desc), 0.0f);
        gtk_label_set_line_wrap(GTK_LABEL(desc), TRUE);
        dt_gui_box_add(GTK_BOX(col), desc);
      }

      dt_gui_box_add(GTK_BOX(box), col);
    }
  }

  return box;
}

// ── public API ────────────────────────────────────────────────────────────────

dt_welcome_screen_t *dt_welcome_screen_new(void)
{
  dt_welcome_screen_t *ws = g_new0(dt_welcome_screen_t, 1);
  ws->pages = g_ptr_array_new_with_free_func(_free_page);
  return ws;
}

int dt_welcome_screen_add_page(dt_welcome_screen_t *ws)
{
  _dt_page_t *pg = g_new0(_dt_page_t, 1);
  pg->questions = g_ptr_array_new_with_free_func(_free_question);
  g_ptr_array_add(ws->pages, pg);
  return (int)ws->pages->len - 1;
}

static void _page_add_checkbox(dt_welcome_screen_t *ws,
                               int page_idx,
                               const char *label,
                               const char *description,
                               gboolean default_active,
                               _dt_welcome_action_t action)
{
  _dt_page_t *pg = ws->pages->pdata[page_idx];

  _dt_question_t *q = g_new0(_dt_question_t, 1);
  q->label = g_strdup(label);
  q->description = g_strdup(description ? description : "");
  q->default_active = default_active;
  q->action = action;
  if(q->action.type != DT_WELCOME_ACTION_NONE)
    q->action.key = g_strdup(action.key);

  g_ptr_array_add(pg->questions, q);
}

void dt_welcome_screen_page_add_dirchooser(dt_welcome_screen_t *ws,
                                           int page_idx,
                                           const char *label,
                                           const char *description,
                                           const char *conf_key)
{
  g_return_if_fail(ws && page_idx >= 0 && page_idx < (int)ws->pages->len);

  _dt_page_t *pg = ws->pages->pdata[page_idx];

  _dt_question_t *q = g_new0(_dt_question_t, 1);
  q->qtype = DT_QUESTION_DIRCHOOSER;
  q->label = g_strdup(label);
  q->description = g_strdup(description ? description : "");
  q->conf_key = g_strdup(conf_key);

  g_ptr_array_add(pg->questions, q);
}

static void _page_add_combobox(dt_welcome_screen_t *ws,
                               int page_idx,
                               const char *label,
                               const char *description,
                               const char *conf_key,
                               const char *const *options,
                               int n_options)
{
  _dt_page_t *pg = ws->pages->pdata[page_idx];
  _dt_question_t *q = g_new0(_dt_question_t, 1);
  q->qtype = DT_QUESTION_COMBOBOX;
  q->label = g_strdup(label);
  q->description = g_strdup(description ? description : "");
  q->conf_key = g_strdup(conf_key);
  q->n_options = n_options;
  q->options = g_new(char *, n_options);
  q->option_labels = g_new(char *, n_options);
  for(int i = 0; i < n_options; i++)
  {
    // split on '|': before is the id (conf value), after is the label
    // (translatable display text). entries without '|' store NULL label
    // and fall back to showing the id
    const char *sep = strchr(options[i], '|');
    if(sep)
    {
      q->options[i] = g_strndup(options[i], sep - options[i]);
      q->option_labels[i] = g_strdup(sep + 1);
    }
    else
    {
      q->options[i] = g_strdup(options[i]);
      q->option_labels[i] = NULL;
    }
  }
  g_ptr_array_add(pg->questions, q);
}

static void _page_add_theme(dt_welcome_screen_t *ws,
                            int page_idx,
                            const char *label,
                            const char *description)
{
  _dt_page_t *pg = ws->pages->pdata[page_idx];
  _dt_question_t *q = g_new0(_dt_question_t, 1);
  q->qtype = DT_QUESTION_THEME;
  q->label = g_strdup(label);
  q->description = g_strdup(description ? description : "");
  g_ptr_array_add(pg->questions, q);
}

// Parse confgen DT_VALUES string "[A][B][C]" into a g_new'd array of g_strdup'd strings.
// *n_out receives the count.  Caller must g_strfreev() the result.
static char **_parse_enum_values(const char *enum_values, int *n_out)
{
  GPtrArray *arr = g_ptr_array_new();
  for(const char *p = enum_values ? enum_values : ""; *p == '[';)
  {
    p++; // skip '['
    const char *end = strchr(p, ']');
    if(!end)
      break;
    g_ptr_array_add(arr, g_strndup(p, end - p));
    p = end + 1;
  }
  *n_out = (int)arr->len;
  g_ptr_array_add(arr, NULL); // NULL-terminate for g_strfreev
  return (char **)g_ptr_array_free(arr, FALSE);
}

void dt_welcome_screen_page_add_conf(dt_welcome_screen_t *ws,
                                     int page_idx,
                                     const char *conf_key,
                                     const char *description)
{
  g_return_if_fail(ws && page_idx >= 0 && page_idx < (int)ws->pages->len);
  g_return_if_fail(conf_key);

  const dt_confgen_type_t type = dt_confgen_type(conf_key);
  const char *label = _(dt_confgen_get_label(conf_key));

  switch(type)
  {
  case DT_BOOL:
  {
    const gboolean def = dt_confgen_get_bool(conf_key, DT_DEFAULT);
    _dt_welcome_action_t action = {DT_WELCOME_ACTION_SET_BOOL, conf_key};
    _page_add_checkbox(ws, page_idx, label, description, def, action);
    break;
  }
  case DT_ENUM:
  {
    int n = 0;
    char **opts = _parse_enum_values(dt_confgen_get(conf_key, DT_VALUES), &n);
    if(n > 0)
      _page_add_combobox(ws, page_idx, label, description,
                         conf_key, (const char *const *)opts, n);
    g_strfreev(opts);
    break;
  }
  case DT_PATH:
    dt_welcome_screen_page_add_dirchooser(ws, page_idx, label, description, conf_key);
    break;
  case DT_STRING:
  {
    // string-type keys can advertise a curated option list to the welcome
    // screen via <welcomescreen options="a,b,c"/> without needing an <enum>
    const char *const *opts = dt_confgen_get_welcome_options(conf_key);
    int n = 0;
    if(opts)
      while(opts[n]) n++;
    if(n > 0)
      _page_add_combobox(ws, page_idx, label, description, conf_key, opts, n);
    else
      dt_print(DT_DEBUG_ALWAYS,
               "[welcome] string conf key '%s' has no welcome options\n", conf_key);
    break;
  }
  default:
    dt_print(DT_DEBUG_ALWAYS,
             "[welcome] conf key '%s' has unsupported type for welcome screen\n", conf_key);
    break;
  }
}

void dt_welcome_screen_page_add_paragraph(dt_welcome_screen_t *ws,
                                          int page_idx,
                                          const char *text,
                                          gboolean centered)
{
  g_return_if_fail(ws && page_idx >= 0 && page_idx < (int)ws->pages->len);

  _dt_page_t *pg = ws->pages->pdata[page_idx];
  _dt_question_t *q = g_new0(_dt_question_t, 1);
  q->qtype = DT_QUESTION_PARAGRAPH;
  q->label = g_strdup(text ? text : "");
  q->centered = centered;
  g_ptr_array_add(pg->questions, q);
}

// Called by the close button; destroys the window, which causes the
// GMainLoop in dt_welcome_screen_show() to quit.
static void _on_close(GtkWidget *btn, gpointer data)
{
  (void)btn;
  gtk_widget_destroy(GTK_WIDGET(data));
}

void dt_welcome_screen_show(dt_welcome_screen_t *ws)
{
  GtkWindow *main_win = GTK_WINDOW(dt_ui_main_window(darktable.gui->ui));

  // Use a plain GtkWindow instead of GtkDialog so there is no
  // built-in (empty) action area leaving dead space at the bottom.
  GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  gtk_window_set_title(GTK_WINDOW(window), _("Welcome to Libre DT-Lab!"));
  gtk_window_set_transient_for(GTK_WINDOW(window), main_win);
  gtk_window_set_modal(GTK_WINDOW(window), TRUE);
  gtk_window_set_destroy_with_parent(GTK_WINDOW(window), TRUE);
  gtk_widget_set_name(window, "welcome-dialog");

#ifdef GDK_WINDOWING_QUARTZ
  dt_osx_disallow_fullscreen(window);
#endif

  gtk_window_set_default_size(GTK_WINDOW(window), 500, -1);
  gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER_ON_PARENT);

  GtkWidget *content = dt_gui_vbox();
  gtk_widget_set_name(content, "welcome-content");
  gtk_container_add(GTK_CONTAINER(window), content);

  // ── static header (logo + app name) – never moves between pages ──────────
  dt_gui_box_add(GTK_BOX(content), _get_logo());

  GtkWidget *app_name = gtk_label_new(_("Welcome to libre-dt-lab!"));
  gtk_widget_set_name(app_name, "welcome-title");
  gtk_widget_set_halign(app_name, GTK_ALIGN_CENTER);
  dt_gui_box_add(GTK_BOX(content), app_name);

  GtkWidget *sep_header = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_widget_set_name(sep_header, "welcome-separator");
  dt_gui_box_add(GTK_BOX(content), sep_header);

  // ── stack of pages ────────────────────────────────────────────────────────
  GtkWidget *stack = gtk_stack_new();
  gtk_stack_set_transition_type(GTK_STACK(stack), GTK_STACK_TRANSITION_TYPE_SLIDE_LEFT_RIGHT);
  gtk_widget_set_name(stack, "welcome-stack");

  const int n_total = (int)ws->pages->len;

  for(int i = 0; i < n_total; i++)
  {
    _dt_page_t *pg = ws->pages->pdata[i];
    GtkWidget *page = _build_page_widget(pg);
    char name[16];
    g_snprintf(name, sizeof(name), "page%d", i);
    gtk_stack_add_named(GTK_STACK(stack), page, name);
  }

  gtk_widget_set_vexpand(stack, TRUE);
  dt_gui_box_add(GTK_BOX(content), dt_gui_expand(stack));

  // ── footer ────────────────────────────────────────────────────────────────
  GtkWidget *sep_footer = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_widget_set_name(sep_footer, "welcome-separator");
  dt_gui_box_add(GTK_BOX(content), sep_footer);

  GtkWidget *footer = dt_gui_hbox();
  gtk_widget_set_name(footer, "welcome-footer");
  dt_gui_box_add(GTK_BOX(content), footer);

  // Progress dots (left-aligned)
  GtkWidget *progress_box = dt_gui_hbox();
  gtk_widget_set_name(progress_box, "welcome-progress");
  gtk_widget_set_halign(progress_box, GTK_ALIGN_START);

  for(int i = 0; i < n_total; i++)
  {
    GtkWidget *dot = gtk_label_new(i == 0 ? "●" : "○");
    gtk_widget_set_name(dot, "welcome-dot");
    dt_gui_box_add(GTK_BOX(progress_box), dot);
  }

  // Navigation buttons (right-aligned)
  GtkWidget *btn_prev = gtk_button_new_with_mnemonic(_("_prev"));
  gtk_widget_set_name(btn_prev, "welcome-prev");

  GtkWidget *btn_next = gtk_button_new_with_mnemonic(_("_next"));
  gtk_widget_set_name(btn_next, "welcome-next");

  GtkWidget *btn_close = gtk_button_new_with_mnemonic(_("_close"));
  gtk_widget_set_name(btn_close, "welcome-close");

  dt_gui_box_add(GTK_BOX(footer), dt_gui_expand(progress_box),
                 btn_prev, btn_next, btn_close);

  // ── navigation state ──────────────────────────────────────────────────────
  _nav_t nav = {
      .stack = stack,
      .prev_btn = btn_prev,
      .next_btn = btn_next,
      .progress_box = progress_box,
      .n_stack_pages = n_total,
      .current = 0,
  };

  g_signal_connect(G_OBJECT(btn_prev), "clicked", G_CALLBACK(_on_prev), &nav);
  g_signal_connect(G_OBJECT(btn_next), "clicked", G_CALLBACK(_on_next), &nav);
  g_signal_connect(G_OBJECT(btn_close), "clicked", G_CALLBACK(_on_close), window);

  // Set initial state (hides prev on page 0)
  _update_navigation(&nav);

  // Block until the window is destroyed (close button or WM close).
  GMainLoop *loop = g_main_loop_new(NULL, FALSE);
  g_signal_connect_swapped(G_OBJECT(window), "destroy",
                           G_CALLBACK(g_main_loop_quit), loop);

  gtk_widget_show_all(window);
  g_main_loop_run(loop);
  g_main_loop_unref(loop);
}

void dt_welcome_screen_free(dt_welcome_screen_t *ws)
{
  if(!ws)
    return;
  g_ptr_array_free(ws->pages, TRUE);
  g_free(ws);
}

// ── content definition & entry point ─────────────────────────────────────────

static dt_welcome_screen_t *_build_welcome_screen(void)
{
  dt_welcome_screen_t *ws = dt_welcome_screen_new();

  // ── Page 0: intro (hardcoded) ─────────────────────────────────────────────
  int page_idx = dt_welcome_screen_add_page(ws);

  dt_welcome_screen_page_add_paragraph
    (ws, page_idx, _("let's set up a few options to get you started."), TRUE);
  dt_welcome_screen_page_add_paragraph
    (ws, page_idx,
     _("if you are unsure, don't worry! just go ahead with the default, "
       "you can always change all values later in the <b>preferences</b>."),
     TRUE);
  dt_welcome_screen_page_add_paragraph
    (ws, page_idx,
     _("you can close this window at any time, in which case the defaults will be applied."),
     TRUE);

  // ── Dynamic conf pages (driven from darktableconfig.xml.in) ──────────────
  GList *keys = dt_confgen_get_welcome_keys();
  int cur_page = 0;

  for(GList *l = keys; l; l = l->next)
  {
    const char *key = l->data;
    const int pagenum = dt_confgen_get_welcome_pagenum(key);
    const char *raw_desc = dt_confgen_get_tooltip(key);
    const char *desc = (raw_desc && *raw_desc) ? _(raw_desc) : NULL;

    if(pagenum != cur_page)
    {
      cur_page = pagenum;
      page_idx = dt_welcome_screen_add_page(ws);
    }

    if(dt_confgen_get_welcome_dirchooser(key))
      dt_welcome_screen_page_add_dirchooser
        (ws, page_idx, _(dt_confgen_get_label(key)), desc, key);
    else if(!g_strcmp0(key, "ui_last/theme"))
      _page_add_theme(ws, page_idx, _(dt_confgen_get_label(key)), desc);
    else
      dt_welcome_screen_page_add_conf(ws, page_idx, key, desc);
  }

  g_list_free_full(keys, g_free);

  // ── Last page: outro (hardcoded) ──────────────────────────────────────────
  page_idx = dt_welcome_screen_add_page(ws);

  dt_welcome_screen_page_add_paragraph
    (ws, page_idx, _("<b>You are all set!</b>"), TRUE);
  dt_welcome_screen_page_add_paragraph
    (ws, page_idx,
     _("libre-dt-lab is extremely configurable and has a wealth of options. "
       "Here we covered just the most essential ones — make sure to explore the "
       "<b>preferences</b> to get the full picture."),
     TRUE);

  {
    char *manual_url = dt_get_manual_url(NULL);

    /* xgettext: %s is a URL; keep <a href="%s"> and </a> unchanged */
    gchar *text = g_strdup_printf(_("Read the <a href=\"%s\">user manual</a> "
                                    "to learn more and make the most of libre-dt-lab."),
                                  manual_url);
    dt_welcome_screen_page_add_paragraph(ws, page_idx, text, TRUE);
    g_free(text);
    g_free(manual_url);
  }

  dt_welcome_screen_page_add_paragraph
    (ws, page_idx,
     _("And remember that you can always click on the <b>?</b> on "
       "the top toolbar to get help right where you need it."),
     TRUE);

  dt_welcome_screen_page_add_paragraph
    (ws, page_idx, _("Happy editing with libre-dt-lab!"), TRUE);

  return ws;
}

void dt_welcome_screen_run_if_needed(void)
{
  if(dt_gimpmode())
    return;
  if(!dt_conf_get_bool("ui/show_welcome_screen"))
    return;

  // Clear the flag before showing so that a crash during the dialog
  // doesn't cause an infinite loop on next launch.
  dt_conf_set_bool("ui/show_welcome_screen", FALSE);

  dt_welcome_screen_t *ws = _build_welcome_screen();
  dt_welcome_screen_show(ws);
  dt_welcome_screen_free(ws);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
