# Agent : contrats techniques et flux d’entraînement

## Architecture et provenance

| Module | Responsabilité |
|---|---|
| `world_observer.py` | Assemble les observations ROS en snapshots NDJSON. |
| `gazebo_pose_source.py` | Fournit uniquement les poses des cartons du simulateur, comme substitut explicite de la caméra externe. |
| `world_encoder.py` | Encode les caractéristiques v2 en ensembles de cartons et obstacles, puis applique une attention conditionnée par le robot. |
| `policy.py` | Politique hybride masquée : logits d’action et paramètres continus de déplacement. |
| `action_executor.py` / `run_agent.py` | Exécution ROS asynchrone et journalisation de l’utilisation live. |
| `demonstrations.py` / `train_targets.py` | Cibles du pupitre et entraînement supervisé commun. |
| `rl/environment.py` / `rl/collect.py` | Match à deux robots, collecte stochastique, résultats et récompenses. |
| `records.py` / `rl/prepare_targets.py` | Journaux d’épisodes bruts puis distributions cibles self-play. |

Les cartons viennent de la caméra externe, ici remplacée explicitement par les
poses des seuls modèles Gazebo `pierre_*`. La pose de notre robot vient de
`/pose_estimation`, les obstacles de `/obstacles_scan` et les tags de
`/aruco_markers`. Vitesse, angle du poignet et états des ventouses proviennent
respectivement de `/odometry/filtered`, `/joint_states` et des topics
`/robot_omni/ventouse_<n>/etat` lorsqu’ils sont reliés à ROS. Sans bridge, les
états des ventouses restent inconnus. Aucun repli vers la pose vraie du robot
n’est autorisé.
Le backend RL fournit séparément la vérité `world` et les `observations` de
chaque robot. La vérité sert aux points et aux journaux, jamais à reconstruire
une observation manquante. Pour une politique, l’adversaire est un obstacle
ponctuel ; son cap, sa charge et sa ventouse sont exclus.

Les mesures périmées restent inconnues. Les ArUco sont transformés depuis
`camera_optical` avec les extrinsèques de l’URDF ; une association ambiguë ne
fabrique pas d’identité de carton. Le décalage temporel pose/image est contrôlé.
L’assembleur ne déduit pas une charge vide de l’absence de mesure.

Les états des trois ventouses gardent leur valeur et leur âge séparément. Un état
inconnu ne vaut ni ventouse libre ni charge vide. `payload_state`, `held_object`
et les contacts ne sont pas déduits des états d’actionneur.

## Contrat des snapshots

Toutes les positions et dimensions sont en mètres dans le même repère fixe de
l'arène (`map`), les angles `yaw` sont en radians. `sequence` augmente à chaque
snapshot et `timestamp` est le temps de référence en secondes. Le producteur
perception doit transformer les sources ROS dans ce repère et fournir des pistes
stables : les IDs ArUco partagés ne sont pas des IDs de carton.

Chaque ligne est un snapshot `schema_version: 2`, dans le repère absolu `map` :

- `sequence`, `timestamp`, `arena.bounds` (`x_min`, `x_max`, `y_min`, `y_max`).
- `observer_monotonic_s` est un temps monotone local utilisé par le runner pour
  détecter les lignes bloquées dans le pipe ; l’observateur et le runner doivent
  tourner sur la même machine.
- `robot.pose` (`x`, `y`, `z`, `yaw`) estimée, `velocity` en `map`, leur âge,
  `wrist_angle_rad` et `suction_states` indexé de 1 à 3. Chaque champ facultatif
  peut rester inconnu lorsqu’aucune source valide ne l’a mesuré.
- `cartons`, `obstacles` : listes d'entités avec un `id` interne
  stable, `pose`, `size`, `confidence`, `age_s`, `visible` et `accessible`
  quand disponibles. Un carton donne aussi `posture` (`vertical` ou `horizontal`), `marker_id` et
  `approach_pose` ; `camera_distance` (ou `distance`) peut décrire la portée
  mesurée d'un tag visible. Le centrage ROS ne dépend pas de cette association
  à un carton. Ce tag reste une observation, pas l'identité du carton.
- `other_robots` reste accepté ; son cercle est ajouté à l’ensemble des obstacles.
- `aruco_detections` : détections locales transformées en `map`, avec leur âge,
  normale de face `normal` (`x,y,z`), profondeur optique `camera_depth`,
  portée et carton associé si l'association est unique. Les détections ambiguës
  restent présentes avec `associated_carton_id: null` et ne rendent aucun carton
  `visible`.
- `zones` : listes de rectangles avec `id`, `center` ou `pose`, `yaw`, `length`,
  `width`, `capacity`, `occupied`, `release_allowed`, `accessible` et
  `approach_pose`. `yaw` tourne le rectangle dans le repère de l'arène.
  Ces champs restent descriptifs et ne sont pas encodés par la politique.
- `mission` : `total_s` (90 s par défaut), `remaining_s`, score et compteurs de
  cartons quand disponibles. L'observateur démarre le compte à rebours au premier
  snapshot horodaté par `/clock`; le temps simulé s'arrête lorsque Gazebo est en pause.
- `active_operation` et `last_action_result` : type, paramètres de déplacement,
  statut, échec et `cancelable`. Les retours exploités par la politique sont
  limités à ces statuts ; aucune progression d’exécution n’est encodée.

Les tableaux doivent contenir toutes les pistes pertinentes, et non uniquement
la cible la plus proche. Une donnée inconnue reste inconnue : ne pas publier un
robot chargé comme vide ni une zone comme libre en l'absence de mesure.
`accessible` doit refléter une vraie vérification de l'approche et du trajet,
pas seulement la présence d'une pose cible.

### Encodage et normalisation

Le schéma exige `schema_version: 2`. Les positions absolues sont normalisées
par les bornes de la carte ; les angles sont représentés par sinus/cosinus.
Chaque carton et obstacle reçoit un embedding 32D. Une attention à quatre têtes,
conditionnée par le robot et le contexte, agrège les deux ensembles sans
dépendre de l’ordre des entités ni tronquer leur nombre. Les inconnues gardent
un indicateur de validité et leur âge lorsqu’il est mesuré. Les IDs et les
détections ArUco ne sont pas des caractéristiques apprises. La fraîcheur et
l'âge des ensembles cartons et obstacles sont encodés même quand ils sont vides.
Un `MOVE` reste autorisé lorsque la perception des obstacles est périmée ; cette
incertitude est disponible comme caractéristique, pas comme filtre d'action.
Les vitesses linéaires sont divisées par une échelle configurable
(`--velocity-scale`, défaut 1 m/s), sans écrêtage. L’encodeur conserve les
coordonnées absolues `map`; une pose inconnue masque `MOVE` sans bloquer les
autres décisions.

## Politique et exécution

La politique possède une tête catégorielle, une tête gaussienne à quatre
variables latentes pour le déplacement et une tête de valeur dont les sorties
sont enregistrées avec les décisions. L’entraînement commun ne l’optimise pas. Les variables
`x,y` sont sigmoïdées dans les limites de la carte réduites par la marge robot
(défaut 20 cm, configurable par `--robot-margin`) ; la sortie contient le couple
`sin θ, cos θ`, normalisé puis décodé avec `atan2`. En collecte, la distribution
d’action masquée utilisée au choix est journalisée pour préparer le self-play.
Les checkpoints conservent la marge et la fraîcheur de pose utilisées au
décodage. Sans option explicite, le runner et le trainer restaurent ces valeurs
du checkpoint. Une substitution via `--robot-margin` ou `--pose-freshness` est
signalée. Les formats de checkpoint inconnus sont refusés ; seuls les checkpoints
v2 sont chargés.

| Action | Commande et conditions |
|---|---|
| `IDLE` | N’émet aucun ordre ; l’opération active continue. |
| `CANCEL` | Demande l’annulation du déplacement via `/go_to_pose` et attend son résultat terminal. |
| `MOVE` | Envoie une cible absolue `x, y, theta` à l’action ROS `/go_to_pose`. Les cibles sans déplacement sont refusées car le contrôleur actuel divise par la longueur du trajet. |
| `SUCTION_<n>_ON/OFF` | Active ou désactive une ventouse précise via son service `SetBool`. |
| `SUCTION_ALL_ON/OFF` | Commande les trois ventouses via `/robot_omni/ventouse/set`. |
| `TOGGLE_WRIST` | Bascule le poignet entre 0° et 90° selon l’angle mesuré. |

L’exécuteur utilise un client d’action ROS `GoToPose` et des clients `SetBool`
sérialisés. L’acceptation, le refus et le résultat terminal du déplacement viennent
du serveur `/go_to_pose`. Un timeout demande l’annulation et conserve le verrou
`unknown` jusqu’au résultat terminal. `CANCEL` arrête les roues côté contrôleur,
puis le serveur confirme l’annulation. Les services ventouse et poignet ne sont
pas annulables. Aucun nouvel ordre n’est envoyé avant une fin confirmée.
Le runner, le trainer et le collecteur partagent le seuil de fraîcheur de pose
du checkpoint ; le backend RL applique aussi ce seuil et la même distance
minimale `MOVE` (0,1 mm). En self-play, chaque robot peut conserver les
paramètres de son propre checkpoint.
Le runner surveille le flux même lorsque `readline()` n'a reçu aucune ligne.
Après `--snapshot-timeout` (1 s par défaut), il verrouille les décisions,
demande l’annulation d’un `MOVE` et ne reprend qu'avec une pose fraîche et un
résultat terminal confirmé.
Le pas d’observation/décision live par défaut est 10 Hz. Après une seconde sans
snapshot frais (`--snapshot-timeout`), le runner suspend les décisions, annule un
`MOVE` actif et ne reprend qu'après une pose fraîche et un résultat terminal
connu. L'horodatage monotone de l'observateur repère aussi les snapshots déjà
accumulés dans le pipe.

## Déroulement d’un épisode RL

| Étape | Traitement |
|---|---|
| 1. Reset | Le backend donne vérité et observations distinctes ; l’environnement crée un ID de partie et garde le reset en mémoire. |
| 2. Observation | Chaque vue reste dans le repère absolu `map`. Le temps total de l’épisode est ajouté. |
| 3. Décision | Les deux politiques choisissent à partir de leur propre vue. Actions, probabilités et valeurs de collecte sont conservées. |
| 4. Exécution | La cible absolue `MOVE` est validée dans les bornes de déplacement, puis envoyée au backend. |
| 5. Retour | `advance` fournit nouvelle vérité, nouvelles observations et réponses d’exécution. Les points d’action sont calculés depuis ces résultats. |
| 6. Fin | Signal explicite de fin ou limite de 90 secondes par défaut ; le gagnant est déterminé depuis les scores vrais ou un `winner_id` explicite. |
| 7. Récompenses | Les points terminaux sont ajoutés à la dernière transition et les récompenses disponibles sont conservées dans le journal brut. |
| 8. Sauvegarde | Le journal d’épisode est écrit après finalisation ; `rl/prepare_targets.py` calcule ensuite les retours et distribue les cibles canoniques. |

Le backend reçoit `max_game_seconds`, le budget restant. Dépasser cette durée,
reculer dans le temps ou ne faire avancer ni l’horloge ni la fin du match est une
erreur. Les séquences périmées et les cibles de navigation ambiguës sont rejetées
avant l’envoi. Les observations ne sont jamais remplacées par la vérité.
Une erreur de step clôt l’épisode comme incomplet ; il faut un reset avant une
nouvelle tentative, car le backend a pu avancer malgré l’erreur.

### Interface du backend et récompenses

La fabrique indiquée à `collect` renvoie un objet avec `reset()` et
`advance(actions, *, max_game_seconds)`, et éventuellement `close()`.
`reset()` et `advance()` renvoient la structure suivante ; `responses` est
facultatif au reset :

```python
{
    "world": {"game_time_s": 0.0, "robots": [...], "scores": {"bleu": 0, "rouge": 0}},
    "observations": {"bleu": snapshot_bleu, "rouge": snapshot_rouge},
    "responses": {"bleu": resultat_bleu, "rouge": resultat_rouge},
}
```

Les deux IDs restent stables. Les snapshots de départ sont dans `map`.
`world.match.finished` ou `world.match_finished` peut terminer le match avant
l’échéance. Les égalités explicites utilisent `winner_id: "draw"` ou `"tie"` ;
sinon les scores finaux permettent de les reconnaître. Un résultat absent reste
inconnu, il n’est pas assimilé à une égalité.

Les fonctions volontairement non implémentées restent :

- `assign_action_points(previous_world, actions, responses, next_world, robot_ids)` ;
- `assign_winner_points(winner_id, robot_ids)`.

Elles doivent retourner une valeur finie par robot. Les actions données à la
fonction de points sont celles transmises au backend, avec leurs poses `map`.
La récompense terminale s’ajoute aux points de la dernière action. Pour un usage
Python, les fonctions peuvent aussi être injectées au constructeur de
`TwoRobotArenaEnv`. Sans elles, `rewards_pending` reste explicite et les fichiers
ne deviennent pas des cibles self-play. Le mode `collect --record-only` est prévu
pour cette situation ; aucun barème de production n’est inventé.

## Fichiers externes et finalisation des cibles

Par défaut, le runner écrit `runs/agent/<episode_id>.jsonl` et la collecte RL
`runs/episodes/<episode_id>.jsonl`. Un chemin explicite finissant par `.jsonl`
ajoute les épisodes finalisés à ce fichier. Un répertoire produit un fichier
par partie, publié par renommage atomique ; il convient donc aussi à plusieurs
processus de collecte. Pendant une partie, les lignes restent en mémoire.
Un arrêt brutal du processus avant finalisation peut perdre cet épisode.

Les lignes RL ont les types `reset`, `decision`, `transition`, `episode_end`.
Une transition conserve notamment :

- `agent_observations` : vues exactes des deux politiques avant la décision ;
- `actions` : décisions et distributions masquées de collecte ; `backend_actions` : poses converties pour le simulateur ;
- `world`, `next_world`, observations brutes et réponses : diagnostic séparé ;
- `reward` : points immédiats, auxquels s’ajoutent les points terminaux sur la dernière transition ;
- `winner_targets` : 1 pour le gagnant, 0 pour le perdant, 0.5 chacun pour une égalité ;
- `complete`, `winner_known`, `rewards_complete`, `episode_id`, `gamma`.

Les cibles de victoire sont propagées sur toutes les lignes, y compris le reset
et les premières décisions, avant toute écriture. Le préparateur calcule les
retours `G[t] = reward[t] + gamma * G[t+1]` à partir des récompenses brutes ; le
défaut de `gamma` est 0.99 par décision. Le `winner_target` est descriptif ; il
ne remplace pas automatiquement les fonctions de points.

Une fermeture ou un reset prématuré finalise un journal `complete: false`, sans
cibles inventées. Une fin sans résultat connu conserve `winner_targets: null`.
Les hooks Python doivent appeler `env.close()` dans un `finally` pour conserver
les épisodes interrompus ; le collecteur CLI le fait automatiquement.

Le runner live enregistre les observations, décisions et résultats disponibles.
Il reconnaît `match.finished`, `match_finished` ou `mission.remaining_s <= 0`.
Une simple fin de fichier ne prouve pas la fin de partie. Ces journaux n’ont pas
de récompenses calculées par le backend et restent des données brutes.

## Cibles et entraînement commun

Le pupitre enregistre le choix au point où son interface appelle le service ou
l’action ROS. Un processus externe ne peut pas observer fiablement les requêtes
de services ni les paramètres des buts d’action sans relais ou introspection
supplémentaire. Chaque choix supporté devient une distribution one-hot sur les
12 actions ; les paramètres `MOVE` sont ceux saisis par l’utilisateur. Le
résultat final reste une métadonnée et ne change jamais l’étiquette. Les `IDLE`
implicites ne sont enregistrés que pendant une opération active.
Une commande de poignet n’est pas convertie en cible si son angle mesuré est
inconnu, car la politique ne peut alors pas déterminer si elle doit basculer.

Les producteurs écrivent le format JSONL commun `arcl-agent-targets-v1`. L’ordre
stable des 12 actions est `IDLE`, `CANCEL`, `MOVE`, `SUCTION_1_ON`,
`SUCTION_2_ON`, `SUCTION_3_ON`, `SUCTION_1_OFF`, `SUCTION_2_OFF`,
`SUCTION_3_OFF`, `TOGGLE_WRIST`, `SUCTION_ALL_ON`, `SUCTION_ALL_OFF` :

- `observation` et `target_action_probs` (12 probabilités dans l’ordre `ACTIONS`) ;
- `target_move_pose` et `move_target_valid` ;
- `group_id`, `sample_id`, `source` et métadonnées de provenance/résultat.

Les journaux self-play contiennent les observations, décisions, distributions
masquées de la politique au moment du choix et récompenses brutes. Le préparateur
`agent.rl.prepare_targets` exige un épisode complet avec récompenses complètes,
calcule `G[t] = reward[t] + gamma * G[t+1]`, puis `r = clip(G / max(|G|), -1, 1)`
par robot et épisode. Il construit la cible
`q = normalize(max(0, p + α·r·(one_hot(action) − p)))`, avec `α = 0.1` par défaut
et configurable à la préparation. Les actions masquées gardent une probabilité
nulle. Les exemples self-play n’entraînent pas les paramètres continus de `MOVE`.

`agent.architecture.train_targets` est l’unique implémentation d’entraînement.
Elle consomme uniquement les cibles canoniques, applique l’entropie croisée à
cibles souples aux logits et les pertes existantes de position/orientation
quand `move_target_valid` est vrai. Le trainer ne lit ni retours, log-probabilités,
avantages ni cibles de valeur.

La séparation train/validation/test se fait par session ou épisode afin de ne
pas répartir un même groupe entre partitions. Au moins deux groupes sont requis
pour une séparation automatique ; les partitions explicites doivent être
disjointes. Le fichier `.metrics.json` conserve les effectifs, sources,
métriques d’action et erreurs de pose. Les checkpoints utilisent le format v2
et restent chargeables par `run_agent --model`. Les anciens formats sont refusés.

Le chargement des cibles vérifie leur compatibilité avec les masques de la
politique avant tout entraînement et signale le fichier et la ligne fautive.
Les métriques comprennent aussi les décisions prises sans opération en cours
(`free_action_samples`, `free_expected_action_accuracy`), les effectifs par
action et le nombre de cibles de déplacement. Une précision sans décision
libre est `null`. Le critère de sélection reste la perte globale de validation.
Si aucune époque n’améliore cette perte, les poids et le statut de la baseline
sont conservés et le rapport indique `best_epoch: 0`. Le checkpoint de sortie
est écrit après la boucle d’entraînement et l’évaluation finale.

## Limites volontairement conservées

La caméra externe réelle, le backend de simulation RL et les règles de points
restent hors périmètre. L'apprentissage actuel utilise les positions des cartons
et les démonstrations `MOVE`/ventouses ; les zones ne sont volontairement pas
encodées. Le centrage vise environ 20 cm entre caméra et tag et ne garantit pas
le contact physique. Les commandes de lancement et d’entraînement sont regroupées
dans [README.md](README.md).
