# Politique de confidentialité — Libre DT-Lab

_Dernière mise à jour : 7 octobre 2026_

Libre DT-Lab est une application de bureau libre et open source (fork
expérimental de darktable). Cette politique décrit ce que l'application fait de
vos informations. En résumé : **Libre DT-Lab ne collecte, ne stocke et ne
transmet aucune donnée personnelle.**

## Données que nous ne collectons pas

Libre DT-Lab n'a **ni télémétrie, ni analyse d'usage, ni rapport de plantage, ni
compte utilisateur**. Nous ne collectons ni votre nom, ni votre e-mail, ni votre
localisation, ni identifiant d'appareil, ni statistiques d'utilisation. Le
développeur ne dispose d'aucun serveur recevant des informations vous
concernant ou concernant votre usage de l'application.

## Données traitées localement

L'application fonctionne entièrement sur votre ordinateur :

* **Vos photos** sont lues dans les dossiers que vous choisissez explicitement
  et ne sont jamais téléversées. Les retouches sont enregistrées dans des
  fichiers annexes (`*.lab.xmp`) et dans une base de données locale.
* **La configuration et le cache** sont stockés localement dans
  `%APPDATA%\libre-dt-lab` (Windows) ou `~/.config/libre-dt-lab` et
  `~/.cache/libre-dt-lab` (Linux/macOS).
* **Les mots de passe** éventuellement mémorisés pour des services de
  publication le sont uniquement dans le magasin d'identifiants du système
  (par ex. le Gestionnaire d'identifiants Windows).

## Accès réseau

Libre DT-Lab se connecte à Internet uniquement pour les fonctions que vous
utilisez explicitement :

* téléchargement des **profils de correction d'objectifs** (données lensfun) ;
* téléchargement de **modèles AI** lorsque vous activez ces fonctions ;
* chargement des **tuiles de carte** (OpenStreetMap) dans la vue carte ;
* connexion aux **destinations de publication** (par ex. une galerie Piwigo)
  que vous configurez vous-même ;
* le tethering appareil photo via USB, qui est une connexion locale.

Ces connexions n'envoient que la requête nécessaire au fonctionnement de la
fonction concernée. Elles ne contiennent aucun identifiant personnel et aucune
donnée d'usage n'est renvoyée au développeur.

## Tiers

Nous ne vendons, ne louons et ne partageons aucune donnée personnelle avec des
tiers. Les téléchargements de profils d'objectifs, de tuiles de carte ou de
modèles AI sont fournis par leurs prestataires publics respectifs, dont les
politiques s'appliquent à ces requêtes.

## Confidentialité des enfants

L'application ne collecte aucune donnée et ne cible pas les enfants.

## Modifications

Toute modification de cette politique sera publiée sur cette page avec une date
de mise à jour.

## Contact

Pour toute question relative à cette politique, ouvrez un ticket sur
<https://github.com/Christian-Bouhon/libre-dt-lab/issues>.

---

English: [Privacy Policy (EN)](privacy-policy.md)
