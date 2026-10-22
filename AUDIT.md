# Audit du dépôt

Ce document recense uniquement les problèmes et les décisions restant à traiter. Les constats liés à l’agent figurent dans la seconde partie.

## Problèmes de paramétrage et de configuration

### RViz : fichier de paramètres mal transmis et absent

Dans le launch, `--params-file` est fourni dans le dictionnaire `parameters`. Il est alors traité comme un nom de paramètre ROS, et non comme un argument de ligne de commande. Par ailleurs, `src/arcl_simulation/config/ekf_params.yaml` n’existe pas dans le dépôt. La configuration visée n’est donc pas chargée.

Référence : [sim.launch.py, ligne 44](src/arcl_simulation/launch/sim.launch.py#L44).

### Dépendances manquantes dans les manifests

- Le CMake de `robot_msgs` exige `action_msgs`, mais [son `package.xml`](src/robot_msgs/package.xml#L10) ne le déclare pas. L’installation des dépendances et l’ordre de compilation peuvent être incomplets dans un environnement propre.
- Le launch d’`arcl_simulation` recherche et démarre le paquet `ros2_aruco`, mais [le `package.xml` du paquet](src/arcl_simulation/package.xml#L10) ne le déclare pas comme dépendance d’exécution. Le launch dépend également de `launch` et `ament_index_python`, importés dans le fichier launch sans dépendances d’exécution explicites.

## Problèmes de logique

### Les commandes de navigation et de centrage se disputent les roues

La boucle de navigation publie des commandes tant qu’un objectif est actif. En parallèle, l’action de centrage ArUco publie sur `/speed_command`, que le même nœud convertit en commandes de roues. Aucune logique ne désactive ou ne priorise l’autre mode : le résultat dépend de l’ordre d’arrivée des publications. Ainsi, l’arrêt publié à la fin du centrage peut être remplacé par la commande de navigation suivante.

Références : [boucle et conversion des commandes](src/arcl_simulation/src/robot_control.cpp#L415), [publication de l’action](src/arcl_simulation/src/rock_centering.cpp#L130).

### Le rayon des obstacles n’est pas pris en compte dans l’évitement

Les messages d’obstacle contiennent un rayon, mais la vérification de conflit compare uniquement la distance au centre avec `robot_radius`. Le calcul des points de détour utilise lui aussi `robot_radius + secu_marge`, sans ajouter le rayon de l’obstacle. Pour les obstacles larges, la trajectoire peut donc passer trop près ou entrer en collision.

Références : [vérification des conflits](src/arcl_simulation/src/robot_control.cpp#L315), [génération du détour](src/arcl_simulation/src/robot_control.cpp#L351).

### Un objectif à la position actuelle peut produire des vitesses non finies

`create_trajectory()` divise par la distance à l’objectif sans traiter le cas où cette distance vaut zéro. Le profil de trajectoire peut alors obtenir une durée NaN; l’objectif reste actif et la boucle de contrôle peut publier des vitesses NaN aux roues.

Références : [calcul de direction](src/arcl_simulation/src/robot_control.cpp#L334), [boucle de contrôle](src/arcl_simulation/src/robot_control.cpp#L415).

### Le centrage est trop loin des cartons

L’action vise une distance caméra-tag de `0,20 m` et déclare le centrage réussi lorsque l’erreur de distance est inférieure à `1 cm`. La détection ArUco repose sur l’identification du quadrilatère du marqueur : si le tag sort du champ ou est trop masqué pour être reconnu, l’action n’a plus de pose récente à suivre et attend, ou finit par expirer.

À la distance de centrage configurée, le robot reste trop loin pour que les ventouses touchent directement le carton. Dans le modèle, le centre optique de la caméra est à environ `x = 0,11 m` et les ventouses avancent jusqu’à `x = 0,147 m` : elles dépassent la caméra d’environ `3,7 cm`. Lorsque le tag est à `0,20 m` de la caméra, il reste donc approximativement `16 cm` entre le plan du tag et les ventouses, sous réserve de l’orientation et de la géométrie de contact. Le succès de l’action signifie que le tag est centré à la distance de référence, pas que l’objet est à portée de préhension.

Trois découpages fonctionnels sont évoqués, sans choix arrêté : intégrer l’approche finale à l’action de centrage ferait de celle-ci une séquence complète jusqu’à la proximité du carton; avancer après le centrage dans le code appelant garderait l’action de centrage limitée à l’alignement, mais imposerait au client d’enchaîner les commandes; revoir l'action pour s'approcher tout en se centrant sur le carton. Le choix dépend notamment de la responsabilité attendue de l’action et de l’endroit où gérer la perte de visibilité du tag pendant l’approche.

Références : [distance et tolérance de centrage](src/arcl_simulation/src/rock_centering.cpp#L43), [contrôle de visibilité et erreur avant/arrière](src/arcl_simulation/src/rock_centering.cpp#L139), [position de la caméra dans l’URDF](src/arcl_simulation/urdf/robot_omni.urdf#L46), [position des ventouses](src/arcl_simulation/worlds/arene_omni.sdf#L119).

## Problèmes peu importants

### La trajectoire continue de défiler pendant une pause de Gazebo

Le nœud `robot_control` n’active pas `use_sim_time`, alors que le launch active cette option pour certains autres nœuds. Comme la trajectoire utilise `now()` et un timer réel, son temps continue à avancer lorsque la simulation est en pause. À la reprise, une trajectoire suffisamment longue peut être déjà terminée et le robot s’arrête.

Ce point est de faible importance pour l’usage courant et concerne surtout les pauses prolongées pendant une trajectoire.

Références : [démarrage du nœud](src/arcl_simulation/launch/sim.launch.py#L48), [départ et durée de trajectoire](src/arcl_simulation/src/robot_control.cpp#L389), [calcul du temps écoulé](src/arcl_simulation/src/robot_control.cpp#L418).

### Utiliser le temps de simulation sur Gazebo et le temps réel sinon

Il faut évidemment utiliser le temps réel sur le robot, mais il serait intéressant d'utiliser le temps de simulation sur Gazebo si jamais on veut entraîner un agent...

## Autre

### Problème aléatoire :
```bash
ros@fedora:/workspaces/arcl_simulation$ ros2 service call /robot_omni/ventouse/set std_srvs/srv/SetBool "{data: true}"   # attraper
waiting for service to become available...
requester: making request: std_srvs.srv.SetBool_Request(data=True)

response:
std_srvs.srv.SetBool_Response(success=False, message="Délai de l'opération dépassé")

ros@fedora:/workspaces/arcl_simulation$ ros2 service call /robot_omni/ventouse/set std_srvs/srv/SetBool "{data: true}"   # attraper
waiting for service to become available...
requester: making request: std_srvs.srv.SetBool_Request(data=True)

response:
std_srvs.srv.SetBool_Response(success=True, message='Objet attaché, montée commandée')

ros@fedora:/workspaces/arcl_simulation$ 
```
Rien n'a changé entre les deux, la simulation était déja lancée depuis un moment, la seule chose qui s'est passée c'est que le robot s'est déplacé. Il faut voir si le problème est dû à la simulation gazebo (cartons contre le mur ou au milieu de la map) ou à autre chose.

---

## Revue de l’agent entraîné — état après corrections

### Éléments encore à traiter

#### AG-01 — P2 — Compléter les démonstrations de prise et de lâcher

Il reste à enregistrer de nouvelles séquences valides de déplacement et de
ventouses afin d’enrichir les groupes d’entraînement.

#### AG-02 — P2 — Évaluer la politique sur des sessions indépendantes

Il reste à constituer des jeux de validation et de test indépendants, puis à
mesurer les prises, lâchers et parties réussis en boucle fermée avant de
modifier le critère de sélection du checkpoint.

#### AG-03 — P2 — Valider l’intégration de bout en bout dans ROS/Gazebo

Il reste à faire un essai live couvrant fin de match, perte d’observations et
commandes dans un workspace ROS/Gazebo configuré.

### Décisions de périmètre

- Les zones restent descriptives dans les snapshots et ne sont volontairement
  pas encodées par le réseau. L’apprentissage par imitation utilise les positions
  des cartons et les démonstrations `MOVE`/ventouses.
- Le backend de self-play Gazebo et les fonctions de récompense ne sont pas inclus
  dans ce chantier. Les données obtenues par imitation restent la voie
  d’apprentissage retenue pour l’instant.
- Les sources réelles de caméra externe, payload et contact confirmé ne sont pas
  encore disponibles ; aucune valeur correspondante n’est déduite des états de
  ventouse seuls.
