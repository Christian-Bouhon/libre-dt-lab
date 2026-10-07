# Privacy Policy — Libre DT-Lab

_Last updated: 7 October 2026_

Libre DT-Lab is a free, open-source desktop application (an experimental fork
of darktable). This policy explains what the application does with your
information. In short: **Libre DT-Lab does not collect, store or transmit any
personal data.**

## Data we do not collect

Libre DT-Lab has **no telemetry, no analytics, no crash reporting and no user
accounts**. We do not collect your name, e-mail, location, device identifiers
or any usage statistics. The developer has no server that receives information
about you or your use of the application.

## Data processed locally

The application works entirely on your own computer:

* **Your photos** are read from the folders you explicitly choose, and are
  never uploaded anywhere. Edits are stored in sidecar files (`*.lab.xmp`) and
  in a local database.
* **Configuration and cache** are stored locally in `%APPDATA%\libre-dt-lab`
  (Windows) or `~/.config/libre-dt-lab` and `~/.cache/libre-dt-lab` (Linux/macOS).
* **Passwords** you may save for publishing services are stored only in the
  operating system's credential store (e.g. Windows Credential Manager).

## Network access

Libre DT-Lab connects to the Internet only for functionality you explicitly
use:

* downloading updated **lens correction profiles** (lensfun data);
* downloading **AI models** when you enable the corresponding features;
* loading **map tiles** (OpenStreetMap) in the map view;
* connecting to **publishing destinations** (for example a Piwigo gallery)
  that you configure yourself;
* camera tethering over USB, which is a local connection.

These connections send only the request needed for the feature to work. They do
not include personal identifiers, and no data about your usage is sent back to
the developer.

## Third parties

We do not sell, rent or share personal data with third parties. Downloads of
lens profiles, map tiles or AI models are served by their respective public
providers, whose own policies apply to those requests.

## Children's privacy

The application does not collect any data and does not target children.

## Changes

Any change to this policy will be published on this page with an updated date.

## Contact

For any question about this policy, open an issue at
<https://github.com/Christian-Bouhon/libre-dt-lab/issues>.

---

Français : [Politique de confidentialité (FR)](privacy-policy.fr.md)
