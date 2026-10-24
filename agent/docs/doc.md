# Contrat de l’agent visuel

Voir [le guide d’utilisation](README.md) pour les commandes.

## Observations et commandes

Les observations conservent l’état structuré en repère `map` : robot,
cartons, obstacles, autres robots, mission, états d’actionneurs, opération active
et dernier résultat. Les mesures inconnues restent inconnues ; chaque source
conserve son âge et sa provenance. La vérité du robot Gazebo ne fournit aucune
caractéristique de politique. Les positions simulées de cartons restent un
fournisseur explicitement sélectionné pour le développement.

`front_camera` contient `path`, `timestamp`, `received_monotonic_s`, `age_s`,
`valid`, `frame_id`, `topic` et la calibration disponible. Les pixels sont stockés
en JPEG qualité 90, hors du JSON, puis décodés en RGB 160 × 120. L’image doit
précéder la commande et avoir au plus 200 ms pour autoriser `SPEED`. L’observateur
émet à 10 Hz ; la caméra simulée fournit 30 Hz.

Le vocabulaire conserve les douze actions précédentes, puis ajoute `SPEED`.
`MOVE` conserve ses quatre sorties latentes pour position et orientation sin/cos.
`SPEED` possède quatre sorties pour magnitude, sin/cos de direction de translation
et vitesse angulaire ; son décodage borne la norme de translation et omega.
Une direction dégénérée utilise l’axe avant. `CENTRAGE` est absent du vocabulaire.

Les embeddings d’entités restent de dimension 32, le contexte structuré de
 dimension 64. Trois convolutions RGB (16/32/64 canaux), une agrégation spatiale
et une fusion avec disponibilité/âge produisent un contexte final de dimension
64. La tête de valeur reste disponible pour les journaux sans perte de valeur
supervisée dans cet entraînement.

## Événements ROS et collecte

Les serveurs publient `/agent/command_events` uniquement lorsque le paramètre de
démarrage en lecture seule `publish_command_events` est vrai (faux par défaut).
Le lancement de simulation expose ce choix avec `command_events:=true`. Hors
activation, aucune télémétrie ni table de suivi de requêtes n’est créée. Le
watchdog des vitesses reste dans la boucle de contrôle existante et fonctionne
indépendamment de la télémétrie. L’observateur vérifie les paramètres des deux
serveurs avant de produire des observations.

Les événements utilisent le format
`arcl-command-event` : `id`, `command`, `status`, `timestamp`, `parameters`,
`message`. Les commandes sont `move`, `speed`, `suction`, `wrist` ; les statuts
signalent acceptation, refus, résultat, annulation ou expiration. Le collecteur
ajoute l’heure monotone de réception et conserve les demandes brutes de vitesse.
Les identifiants permettent de relier commande et résultat.

Le centrage peut publier ses vitesses comme démonstrateur. Le collecteur observe
ces vitesses et leur acceptation ; il ne s’abonne pas à l’action de centrage.
Le pupitre et la stratégie n’ont pas à produire de cibles d’apprentissage.

Les vitesses sont dans `base_link`, en m/s et rad/s. Le contrôleur applique un
watchdog de 200 ms, vérifie les bornes et refuse ces commandes pendant une
navigation absolue. L’exécuteur borne chaque commande apprise à 100 ms et publie
ensuite un arrêt. L’acceptation par les serveurs reste la référence de collecte.

Une session écrit progressivement un manifeste, des observations, des événements
et des images. La préparation prend l’observation strictement antérieure à
l’événement reçu, reconstruit les opérations, ajuste l’âge d’image et limite les
exemples de vitesse à 10 Hz. Les commandes nulles restent des exemples de vitesse.
Les observations pendant une navigation ou un service actif donnent des exemples
`IDLE` ; les flux de vitesse ne produisent pas d’attentes implicites.

## Cibles et apprentissage

`arcl-agent-targets` contient l’observation, les métadonnées de provenance,
les probabilités d’action et les paires `move_target_valid`/`target_move_pose` et
`speed_target_valid`/`target_speed`. Les cibles de vitesse sont stockées dans les
unités physiques `vx`, `vy`, `omega`. La perte apprend leur décodage ainsi que la
magnitude et la direction sin/cos ; la direction n’est pas supervisée à vitesse
nulle. Chaque perte continue est appliquée uniquement à sa cible valide.

La séparation des données s’effectue par session/épisode. L’équilibrage des actions
reste présent. La symétrie du terrain est désactivée sur les exemples qui
référencent une image ; le reflet d’une scène ne permet pas de reconstruire la
vue embarquée correspondante. Les images sont chargées à la demande et traitées
par lots pendant l’entraînement.

Les métriques comprennent perte, précision catégorielle attendue, erreurs de pose,
erreur vectorielle de vitesse linéaire, erreur de vitesse angulaire, effectifs par
action et disponibilité d’image. Chaque run conserve configuration, données et
partitions identifiées, journal, métriques et meilleur checkpoint.

Les checkpoints `arcl-agent-policy` contiennent les poids du réseau courant, les
limites de vitesse et les paramètres de normalisation. Le code utilise uniquement
la structure actuelle, sans versions ni migration. Les contrôles de contenu
restent appliqués aux observations, aux cibles et aux poids chargés.
La préparation self-play met à jour la cible catégorielle et conserve les pertes
continues masquées lorsqu’aucune règle de supervision ne les définit.
