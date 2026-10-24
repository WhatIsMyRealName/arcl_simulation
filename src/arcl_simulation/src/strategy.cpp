#include <chrono>
#include <cmath>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include <robot_msgs/action/go_to_pose.hpp>
#include <robot_msgs/action/rock_centering.hpp>

using namespace std::chrono_literals;

// --- interfaces (les mêmes que strategie_camelot.py) ---
static const std::string ACTION_GO_TO_POSE = "/go_to_pose";
static const std::string ACTION_ROCK_CENTERING = "/rock_centering";
static const std::string SERVICE_TOUTES = "/robot_omni/ventouse/set";      // true = prendre les 3, false = tout lâcher
static const std::string SERVICE_POIGNET = "/robot_omni/poignet/set";      // true = 90° (pierres debout), false = 0°
// un étage : "/robot_omni/ventouse_<n>/set", n = 1..3
static const std::string REPERE = "map";
static const std::string ROBOT = "base_link";

// --- géométrie (arene_omni.sdf) ---
static constexpr double TABLE_X = 3.0;
static constexpr double PIERRE_P = 0.110;
static constexpr double BOUCHE_VENTOUSE = 0.152;   // m : face des ventouses devant le centre du robot
static constexpr double JEU_PRISE = 0.015;         // m : s'arrêter 1 à 2 cm devant la pile
static constexpr double DIST_PILE = BOUCHE_VENTOUSE + JEU_PRISE + PIERRE_P / 2;   // 0.222 m
static constexpr double RECUL_CARRIERE = 0.10;     // m : pose d'approche devant une carrière

// --- réglages ---
static constexpr double T_ALLER = 20.0;     // s : délai max d'un go_to_pose
static constexpr double T_CENTRAGE = 10.0;  // s : délai max d'un rock_centering
static constexpr double T_SERVICE = 10.0;   // s : délai max d'un appel de service
static constexpr int MAX_ESSAIS = 2;        // essais par action avant d'abandonner la séquence


class Strategy : public rclcpp::Node{

        public:
            using GoToPose = robot_msgs::action::GoToPose;
            using GoalHandleGoToPose = rclcpp_action::ClientGoalHandle<GoToPose>;
            using RockCentering = robot_msgs::action::RockCentering;
            using GoalHandleRockCentering = rclcpp_action::ClientGoalHandle<RockCentering>;
            using SetBool = std_srvs::srv::SetBool;

            Strategy() : Node("strategy"){

                duree_match = this->declare_parameter("duree_match", 0.0);   // s, 0 = pas de limite
                bleu = this->declare_parameter("bleu", false);               // true : tout en miroir (x -> 3 - x)
                // après /rock_centering, avance jusqu'à la distance de prise :
                //   < 0 : calculée sur la carte le long du cap corrigé (garde la correction latérale)
                //   0   : /rock_centering s'arrête déjà à la bonne distance
                //   > 0 : avance fixe (m)
                avance_apres_centrage = this->declare_parameter("avance_apres_centrage", -1.0);

                move_to_client_ptr = rclcpp_action::create_client<GoToPose>(this, ACTION_GO_TO_POSE);
                centrage_client_ptr = rclcpp_action::create_client<RockCentering>(this, ACTION_ROCK_CENTERING);

                ventouse_clients[0] = this->create_client<SetBool>(SERVICE_TOUTES);
                for (int etage = 1; etage <= 3; ++etage){
                    ventouse_clients[etage] = this->create_client<SetBool>(
                        "/robot_omni/ventouse_" + std::to_string(etage) + "/set");
                }
                poignet_client = this->create_client<SetBool>(SERVICE_POIGNET);

                tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
                tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

                // Pas de tick() ici : l'exécuteur ne tourne pas encore, aucun callback ne serait reçu.
                // Un timer attend que tous les serveurs soient là, puis lance la séquence.
                attente_timer = this->create_wall_timer(500ms, [this](){ attendre_serveurs(); });

                RCLCPP_INFO(this->get_logger(), "Strategy node launched successfuly (%s)", bleu ? "bleu" : "jaune");
            }


        private:

            enum class State { IDLE, MOVE_TO, CENTRAGE, ROTATION_VENTOUSE, VENTOUSE, FINISH };

            // MOVE_TO           : x, y (m), theta (degrés)
            // CENTRAGE          : x, y = centre de la pile visée. Appelle /rock_centering ; s'il réussit, le MOVE_TO
            //                     qui suit (pose de prise calculée sur la carte) est remplacé par la pose corrigée.
            //                     S'il échoue, ce MOVE_TO est joué tel quel.
            // ROTATION_VENTOUSE : direction 1 = poignet à 90° (pierres debout), 0 = poignet à 0°
            // VENTOUSE          : index 0 = les 3 étages, 1..3 = un seul étage ; direction 1 = prendre, 0 = lâcher
            struct Action{
                State state = State::IDLE;
                float x = 0.0;
                float y = 0.0;
                float theta = 0.0;
                int direction = 0;
                int index = -2;
            };

            // Exemple : carrière bord_nord_1 -> mur_nord, puis retour en salle du trône.
            // Poses calculées comme dans strategie_camelot.py (centre robot à 0.222 m de l'axe des pierres,
            // approche / dégagement 10 cm plus loin). Attention : ici pas d'A*, go_to_pose va en ligne droite.
            /*std::list<Action> actions = {
                {State::IDLE, 0.0, 0.0, 0.0, 0, 0},
                {State::ROTATION_VENTOUSE, 0.0, 0.0, 0.0, 0, 0},     // poignet à 0° (pierres couchées)
                {State::MOVE_TO, 0.75, 1.621, 90.0, 0, 0},           // approche carrière bord_nord_1
                {State::CENTRAGE, 0.75, 1.943, 0.0, 0, 0},           // centrage sur la pile (centre de la carrière)
                {State::MOVE_TO, 0.75, 1.721, 90.0, 0, 0},           // contact (remplacé si le centrage réussit)
                {State::VENTOUSE, 0.0, 0.0, 0.0, 1, 0},              // prendre les 3 pierres
                {State::MOVE_TO, 0.75, 1.621, 90.0, 0, 0},           // dégagement
                {State::MOVE_TO, 0.25, 1.255, 90.0, 0, 0},           // approche mur_nord
                {State::MOVE_TO, 0.25, 1.355, 90.0, 0, 0},           // contact
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 0},              // lâcher les 3 : mur
                {State::MOVE_TO, 0.25, 1.255, 90.0, 0, 0},           // dégagement
                {State::MOVE_TO, 0.25, 1.00, 0.0, 0, 0},             // salle du trône
                {State::FINISH, 0.0, 0.0, 0.0, 0, 0}
            };*/

                        std::list<Action> actions = {
                {State::IDLE, 0.0, 0.0, 0.0, 0, 0},
                {State::ROTATION_VENTOUSE, 0.0, 0.0, 0.0, 0, 0},   // poignet à 0° (pierres couchées)
                {State::MOVE_TO, 0.750, 0.379, -90.0, 0, 0},       // vers carrière bord_sud_1
                {State::CENTRAGE, 0.750, 0.057, 0.0, 0, 0},        // centrage sur bord_sud_1
                {State::MOVE_TO, 0.750, 0.279, -90.0, 0, 0},       // contact carrière (remplacé si le centrage réussit)
                {State::VENTOUSE, 0.0, 0.0, 0.0, 1, 0},            // prendre les 3 pierres
                {State::MOVE_TO, 0.750, 0.379, -90.0, 0, 0},       // dégagement carrière
                {State::MOVE_TO, 0.230, 0.747, -90.0, 0, 0},       // vers mur_sud
                {State::MOVE_TO, 0.230, 0.647, -90.0, 0, 0},       // contact mur_sud
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 0},            // lâcher tout : mur_sud
                {State::MOVE_TO, 0.230, 0.747, -90.0, 0, 0},       // dégagement
                {State::MOVE_TO, 0.750, 1.621, 90.0, 0, 0},        // vers carrière bord_nord_1
                {State::CENTRAGE, 0.750, 1.943, 0.0, 0, 0},        // centrage sur bord_nord_1
                {State::MOVE_TO, 0.750, 1.721, 90.0, 0, 0},        // contact carrière (remplacé si le centrage réussit)
                {State::VENTOUSE, 0.0, 0.0, 0.0, 1, 0},            // prendre les 3 pierres
                {State::MOVE_TO, 0.750, 1.621, 90.0, 0, 0},        // dégagement carrière
                {State::MOVE_TO, 0.806, 1.451, 135.0, 0, 0},       // vers tour_nord_ext
                {State::ROTATION_VENTOUSE, 0.0, 0.0, 0.0, 1, 0},   // poignet à 90° (pierres debout)
                {State::MOVE_TO, 0.735, 1.522, 135.0, 0, 0},       // contact tour_nord_ext
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 1},            // lâcher étage 1 : tour_nord_ext
                {State::MOVE_TO, 0.806, 1.451, 135.0, 0, 0},       // dégagement
                {State::MOVE_TO, 0.825, 1.425, 135.0, 0, 0},       // vers cour
                {State::MOVE_TO, 0.950, 1.000, 146.2, 0, 0},       // vers cour
                {State::MOVE_TO, 0.952, 1.000, 180.0, 0, 0},       // vers cour
                {State::MOVE_TO, 0.852, 1.000, 180.0, 0, 0},       // contact cour
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 0},            // lâcher tout : cour
                {State::MOVE_TO, 0.952, 1.000, 180.0, 0, 0},       // dégagement
                {State::ROTATION_VENTOUSE, 0.0, 0.0, 0.0, 0, 0},   // poignet à 0° (pierres couchées)
                {State::MOVE_TO, 1.200, 1.525, 90.0, 0, 0},        // vers carrière bord_nord_2
                {State::MOVE_TO, 1.250, 1.621, 90.0, 0, 0},        // vers carrière bord_nord_2
                {State::CENTRAGE, 1.250, 1.943, 0.0, 0, 0},        // centrage sur bord_nord_2
                {State::MOVE_TO, 1.250, 1.721, 90.0, 0, 0},        // contact carrière (remplacé si le centrage réussit)
                {State::VENTOUSE, 0.0, 0.0, 0.0, 1, 0},            // prendre les 3 pierres
                {State::MOVE_TO, 1.250, 1.621, 90.0, 0, 0},        // dégagement carrière
                {State::MOVE_TO, 1.156, 1.101, 135.0, 0, 0},       // vers tour_nord_int
                {State::ROTATION_VENTOUSE, 0.0, 0.0, 0.0, 1, 0},   // poignet à 90° (pierres debout)
                {State::MOVE_TO, 1.085, 1.172, 135.0, 0, 0},       // contact tour_nord_int
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 1},            // lâcher étage 1 : tour_nord_int
                {State::MOVE_TO, 1.156, 1.101, 135.0, 0, 0},       // dégagement
                {State::MOVE_TO, 1.175, 1.075, 135.0, 0, 0},       // vers tour_sud_ext
                {State::MOVE_TO, 1.000, 0.825, 135.0, 0, 0},       // vers tour_sud_ext
                {State::MOVE_TO, 0.800, 0.550, 157.5, 0, 0},       // vers tour_sud_ext
                {State::MOVE_TO, 0.806, 0.549, -135.0, 0, 0},      // vers tour_sud_ext
                {State::MOVE_TO, 0.735, 0.478, -135.0, 0, 0},      // contact tour_sud_ext
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 3},            // lâcher étage 3 : tour_sud_ext
                {State::MOVE_TO, 0.806, 0.549, -135.0, 0, 0},      // dégagement
                {State::MOVE_TO, 0.825, 0.575, -135.0, 0, 0},      // vers tour_sud_int
                {State::MOVE_TO, 1.000, 0.575, -146.3, 0, 0},      // vers tour_sud_int
                {State::MOVE_TO, 1.078, 0.522, 135.0, 0, 0},       // vers tour_sud_int
                {State::MOVE_TO, 1.007, 0.593, 135.0, 0, 0},       // contact tour_sud_int
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 2},            // lâcher étage 2 : tour_sud_int
                {State::MOVE_TO, 1.078, 0.522, 135.0, 0, 0},       // dégagement
                {State::ROTATION_VENTOUSE, 0.0, 0.0, 0.0, 0, 0},   // poignet à 0° (pierres couchées)
                {State::MOVE_TO, 1.200, 0.400, -101.3, 0, 0},      // vers carrière bord_sud_2
                {State::MOVE_TO, 1.250, 0.379, -90.0, 0, 0},       // vers carrière bord_sud_2
                {State::CENTRAGE, 1.250, 0.057, 0.0, 0, 0},        // centrage sur bord_sud_2
                {State::MOVE_TO, 1.250, 0.279, -90.0, 0, 0},       // contact carrière (remplacé si le centrage réussit)
                {State::VENTOUSE, 0.0, 0.0, 0.0, 1, 0},            // prendre les 3 pierres
                {State::MOVE_TO, 1.250, 0.379, -90.0, 0, 0},       // dégagement carrière
                {State::MOVE_TO, 0.850, 0.425, -45.0, 0, 0},       // vers mur_nord
                {State::MOVE_TO, 0.550, 0.700, -45.0, 0, 0},       // vers mur_nord
                {State::MOVE_TO, 0.325, 0.925, -45.0, 0, 0},       // vers mur_nord
                {State::MOVE_TO, 0.300, 0.925, -45.0, 0, 0},       // vers mur_nord
                {State::MOVE_TO, 0.225, 1.000, -45.0, 0, 0},       // vers mur_nord
                {State::MOVE_TO, 0.230, 1.253, 90.0, 0, 0},        // vers mur_nord
                {State::MOVE_TO, 0.230, 1.353, 90.0, 0, 0},        // contact mur_nord
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 0},            // lâcher tout : mur_nord
                {State::MOVE_TO, 0.230, 1.253, 90.0, 0, 0},        // dégagement
                {State::MOVE_TO, 0.350, 0.875, -33.8, 0, 0},       // vers carrière centre_nord
                {State::MOVE_TO, 0.625, 0.600, -33.8, 0, 0},       // vers carrière centre_nord
                {State::MOVE_TO, 0.650, 0.600, -33.8, 0, 0},       // vers carrière centre_nord
                {State::MOVE_TO, 0.700, 0.550, -33.8, 0, 0},       // vers carrière centre_nord
                {State::MOVE_TO, 0.850, 0.450, -33.8, 0, 0},       // vers carrière centre_nord
                {State::MOVE_TO, 0.900, 0.450, -33.8, 0, 0},       // vers carrière centre_nord
                {State::MOVE_TO, 1.050, 0.600, -33.8, 0, 0},       // vers carrière centre_nord
                {State::MOVE_TO, 1.150, 0.750, -33.8, 0, 0},       // vers carrière centre_nord
                {State::MOVE_TO, 1.178, 1.495, 0.0, 0, 0},         // vers carrière centre_nord
                {State::CENTRAGE, 1.500, 1.495, 0.0, 0, 0},        // centrage sur centre_nord
                {State::MOVE_TO, 1.278, 1.495, 0.0, 0, 0},         // contact carrière (remplacé si le centrage réussit)
                {State::VENTOUSE, 0.0, 0.0, 0.0, 1, 0},            // prendre les 3 pierres
                {State::MOVE_TO, 1.178, 1.495, 0.0, 0, 0},         // dégagement carrière
                {State::MOVE_TO, 0.975, 1.650, -112.5, 0, 0},      // vers mur_aruco_nord
                {State::MOVE_TO, 0.925, 1.675, -135.0, 0, 0},      // vers mur_aruco_nord
                {State::MOVE_TO, 0.903, 1.653, -135.0, 0, 0},      // vers mur_aruco_nord
                {State::MOVE_TO, 0.832, 1.582, -135.0, 0, 0},      // contact mur_aruco_nord
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 0},            // lâcher tout : mur_aruco_nord
                {State::MOVE_TO, 0.903, 1.653, -135.0, 0, 0},      // dégagement
                {State::MOVE_TO, 1.150, 1.325, 0.0, 0, 0},         // vers carrière centre_sud
                {State::MOVE_TO, 1.178, 0.505, 0.0, 0, 0},         // vers carrière centre_sud
                {State::CENTRAGE, 1.500, 0.505, 0.0, 0, 0},        // centrage sur centre_sud
                {State::MOVE_TO, 1.278, 0.505, 0.0, 0, 0},         // contact carrière (remplacé si le centrage réussit)
                {State::VENTOUSE, 0.0, 0.0, 0.0, 1, 0},            // prendre les 3 pierres
                {State::MOVE_TO, 1.178, 0.505, 0.0, 0, 0},         // dégagement carrière
                {State::MOVE_TO, 0.875, 0.475, -11.3, 0, 0},       // vers mur_aruco_sud
                {State::MOVE_TO, 0.825, 0.475, -22.5, 0, 0},       // vers mur_aruco_sud
                {State::MOVE_TO, 0.800, 0.475, -33.8, 0, 0},       // vers mur_aruco_sud
                {State::MOVE_TO, 0.725, 0.550, -45.0, 0, 0},       // vers mur_aruco_sud
                {State::MOVE_TO, 0.550, 0.700, -45.0, 0, 0},       // vers mur_aruco_sud
                {State::MOVE_TO, 0.447, 0.803, -45.0, 0, 0},       // vers mur_aruco_sud
                {State::MOVE_TO, 0.518, 0.732, -45.0, 0, 0},       // contact mur_aruco_sud
                {State::VENTOUSE, 0.0, 0.0, 0.0, 0, 0},            // lâcher tout : mur_aruco_sud
                {State::MOVE_TO, 0.447, 0.803, -45.0, 0, 0},       // dégagement
                {State::MOVE_TO, 0.325, 0.925, -45.0, 0, 0},       // retour salle du trône
                {State::MOVE_TO, 0.250, 1.000, 0.0, 0, 0},         // retour salle du trône
                {State::FINISH, 0.0, 0.0, 0.0, 0, 0}
            };




            rclcpp_action::Client<GoToPose>::SharedPtr move_to_client_ptr;
            rclcpp_action::Client<RockCentering>::SharedPtr centrage_client_ptr;
            std::map<int, rclcpp::Client<SetBool>::SharedPtr> ventouse_clients;
            rclcpp::Client<SetBool>::SharedPtr poignet_client;

            std::unique_ptr<tf2_ros::Buffer> tf_buffer;
            std::shared_ptr<tf2_ros::TransformListener> tf_listener;

            rclcpp::TimerBase::SharedPtr attente_timer;
            rclcpp::TimerBase::SharedPtr match_timer;
            rclcpp::TimerBase::SharedPtr watchdog;
            rclcpp::TimerBase::SharedPtr arret_timer;

            GoalHandleGoToPose::SharedPtr goal_handle;
            GoalHandleRockCentering::SharedPtr centrage_handle;
            uint64_t etape = 0;      // change à chaque tick : les réponses d'une étape passée sont ignorées
            int essais = 0;
            bool termine = false;
            double duree_match = 0.0;
            bool bleu = false;
            double avance_apres_centrage = -1.0;


            // ------------------------------------------------------------ démarrage et fin

            void attendre_serveurs(){
                std::vector<std::string> manquants;
                if (!move_to_client_ptr->action_server_is_ready()) manquants.push_back(ACTION_GO_TO_POSE);
                if (!centrage_client_ptr->action_server_is_ready()) manquants.push_back(ACTION_ROCK_CENTERING);
                for (auto & [etage, client] : ventouse_clients){
                    if (!client->service_is_ready()) manquants.push_back(client->get_service_name());
                }
                if (!poignet_client->service_is_ready()) manquants.push_back(poignet_client->get_service_name());

                if (!manquants.empty()){
                    std::string liste;
                    for (const auto & m : manquants) liste += (liste.empty() ? "" : ", ") + m;
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "En attente de : %s", liste.c_str());
                    return;
                }

                attente_timer->cancel();
                RCLCPP_INFO(this->get_logger(), "Tous les serveurs sont prêts : c'est parti");
                if (duree_match > 0.0){
                    match_timer = this->create_wall_timer(
                        std::chrono::milliseconds(static_cast<int64_t>(duree_match * 1000.0)),
                        [this](){ fin_de_match(); });
                }
                tick();
            }

            void fin_de_match(){
                match_timer->cancel();
                if (goal_handle) move_to_client_ptr->async_cancel_goal(goal_handle);
                if (centrage_handle) centrage_client_ptr->async_cancel_goal(centrage_handle);
                terminer("Fin du match : arrêt", false);
            }

            void terminer(const std::string & raison, bool erreur){
                if (termine) return;
                termine = true;
                ++etape;
                annuler_watchdog();

                if (erreur) RCLCPP_ERROR(this->get_logger(), "%s", raison.c_str());
                else RCLCPP_INFO(this->get_logger(), "%s", raison.c_str());
                // petit délai pour laisser partir une éventuelle annulation de but
                arret_timer = this->create_wall_timer(500ms, [](){ rclcpp::shutdown(); });
            }


            // ------------------------------------------------------------ machine à états

            void tick(){
                if (termine) return;
                if (actions.empty()){
                    terminer("Plus d'action dans la liste", false);
                    return;
                }
                ++etape;
                annuler_watchdog();

                const Action action = actions.front();

                switch (action.state){
                    case State::FINISH:
                        terminer("Stratégie terminée", false);
                        break;
                    case State::IDLE:
                        RCLCPP_INFO(this->get_logger(), "Je commence à travailler");
                        suivante();
                        break;
                    case State::MOVE_TO:
                        RCLCPP_INFO(this->get_logger(), "Je vais là bas : (%.3f, %.3f, %.0f°)", action.x, action.y, action.theta);
                        send_move_to(action);
                        break;
                    case State::CENTRAGE:
                        RCLCPP_INFO(this->get_logger(), "Je me centre sur la pile (%.3f, %.3f)", action.x, action.y);
                        send_centrage();
                        break;
                    case State::ROTATION_VENTOUSE:
                        RCLCPP_INFO(this->get_logger(), "Je rotate la ventouse à %s", action.direction ? "90°" : "0°");
                        appeler_service(poignet_client, action.direction != 0, "poignet");
                        break;
                    case State::VENTOUSE: {
                        auto it = ventouse_clients.find(action.index);
                        if (it == ventouse_clients.end()){
                            RCLCPP_ERROR(this->get_logger(), "Ventouse %d inconnue (0 = toutes, 1..3 = un étage) : action sautée", action.index);
                            suivante();
                            break;
                        }
                        const std::string nom = action.index == 0 ? "ventouses (toutes)" : "ventouse " + std::to_string(action.index);
                        RCLCPP_INFO(this->get_logger(), "Je %s avec %s", action.direction ? "prends" : "lâche", nom.c_str());
                        appeler_service(it->second, action.direction != 0, nom);
                        break;
                    }
                    default:
                        RCLCPP_ERROR(this->get_logger(), "Cas non défini");
                        suivante();
                        break;
                }
            }

            void suivante(){
                essais = 0;
                actions.pop_front();
                tick();
            }

            void echec(const std::string & raison){
                ++essais;
                if (essais < MAX_ESSAIS){
                    RCLCPP_WARN(this->get_logger(), "%s → nouvel essai (%d/%d)", raison.c_str(), essais + 1, MAX_ESSAIS);
                    tick();     // rejoue la même action
                    return;
                }
                // la suite de la liste dépend de cette étape : on s'arrête plutôt que de continuer au hasard
                terminer("Abandon de la séquence : " + raison, true);
            }


            // ------------------------------------------------------------ délais

            void lancer_watchdog(double secondes, std::function<void()> sur_delai){
                annuler_watchdog();
                const auto id = etape;
                watchdog = this->create_wall_timer(
                    std::chrono::milliseconds(static_cast<int64_t>(secondes * 1000.0)),
                    [this, id, sur_delai](){
                        watchdog->cancel();
                        if (id != etape) return;
                        sur_delai();
                    });
            }

            void annuler_watchdog(){
                if (watchdog){
                    watchdog->cancel();
                    watchdog.reset();
                }
            }


            // ------------------------------------------------------------ go_to_pose

            void send_move_to(const Action & action){
                double x = action.x;
                double theta = action.theta;
                if (bleu){
                    x = TABLE_X - x;
                    theta = 180.0 - theta;
                }

                auto goal = GoToPose::Goal();
                goal.x = x;
                goal.y = action.y;
                goal.theta = theta * M_PI / 180.0;     // la liste est en degrés, go_to_pose attend des radians

                const auto id = etape;

                auto options = rclcpp_action::Client<GoToPose>::SendGoalOptions();
                options.goal_response_callback = [this, id](GoalHandleGoToPose::SharedPtr gh){
                    move_to_goal_response_callback(id, gh);
                };
                options.feedback_callback = [this](GoalHandleGoToPose::SharedPtr gh,
                                                   const std::shared_ptr<const GoToPose::Feedback> feedback){
                    move_to_feedback_callback(gh, feedback);
                };
                options.result_callback = [this, id](const GoalHandleGoToPose::WrappedResult & result){
                    move_to_result_callback(id, result);
                };
                move_to_client_ptr->async_send_goal(goal, options);

                lancer_watchdog(T_ALLER, [this](){
                    if (goal_handle) move_to_client_ptr->async_cancel_goal(goal_handle);
                    else move_to_client_ptr->async_cancel_all_goals();
                    goal_handle.reset();

                    echec("go_to_pose pas fini à temps : annulé");
                });
            }

            void move_to_goal_response_callback(uint64_t id, GoalHandleGoToPose::SharedPtr gh){
                if (id != etape){
                    if (gh) move_to_client_ptr->async_cancel_goal(gh);   // réponse d'une étape déjà abandonnée
                    return;
                }
                if (!gh){
                    RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");

                    echec("go_to_pose refusé par le serveur");
                    return;
                }
                goal_handle = gh;
                RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
            }

            void move_to_feedback_callback(GoalHandleGoToPose::SharedPtr,
                                           const std::shared_ptr<const GoToPose::Feedback> feedback){
                (void)feedback;   // à exploiter selon les champs de GoToPose.action
            }

            void move_to_result_callback(uint64_t id, const GoalHandleGoToPose::WrappedResult & result){
                if (id != etape) return;
                goal_handle.reset();
                switch (result.code){
                    case rclcpp_action::ResultCode::SUCCEEDED:
                        RCLCPP_INFO(this->get_logger(), "Arrivé");

                        suivante();
                        return;
                    case rclcpp_action::ResultCode::ABORTED:
                        RCLCPP_ERROR(this->get_logger(), "Goal was aborted");

                        echec("go_to_pose abandonné");
                        return;
                    case rclcpp_action::ResultCode::CANCELED:
                        RCLCPP_ERROR(this->get_logger(), "Goal was canceled");

                        echec("go_to_pose annulé");
                        return;
                    default:
                        RCLCPP_ERROR(this->get_logger(), "Unknown result code");

                        echec("go_to_pose : code de résultat inconnu");
                        return;
                }
            }


            // ------------------------------------------------------------ rock_centering

            void send_centrage(){
                const auto id = etape;

                auto options = rclcpp_action::Client<RockCentering>::SendGoalOptions();
                options.goal_response_callback = [this, id](GoalHandleRockCentering::SharedPtr gh){
                    centrage_goal_response_callback(id, gh);
                };
                options.feedback_callback = [this](GoalHandleRockCentering::SharedPtr,
                                                   const std::shared_ptr<const RockCentering::Feedback> feedback){
                    centrage_feedback_callback(feedback);
                };
                options.result_callback = [this, id](const GoalHandleRockCentering::WrappedResult & result){
                    centrage_result_callback(id, result);
                };
                centrage_client_ptr->async_send_goal(RockCentering::Goal(), options);   // goal vide : simple déclenchement

                lancer_watchdog(T_CENTRAGE, [this, id](){
                    if (centrage_handle) centrage_client_ptr->async_cancel_goal(centrage_handle);
                    else centrage_client_ptr->async_cancel_all_goals();
                    centrage_handle.reset();
                    fin_centrage(id, false, "pas fini à temps : annulé");
                });
            }

            void centrage_goal_response_callback(uint64_t id, GoalHandleRockCentering::SharedPtr gh){
                if (id != etape){
                    if (gh) centrage_client_ptr->async_cancel_goal(gh);
                    return;
                }
                if (!gh){
                    fin_centrage(id, false, "refusé par le serveur");
                    return;
                }
                centrage_handle = gh;
            }

            void centrage_feedback_callback(const std::shared_ptr<const RockCentering::Feedback> feedback){
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "  centrage %.1f s : marqueur %s, erreurs avant %.3f m, latérale %.3f m, cap %.1f°",
                    feedback->elapsed, feedback->marker_visible ? "vu" : "perdu",
                    feedback->err_fwd, feedback->err_lat, feedback->err_theta * 180.0 / M_PI);
            }

            void centrage_result_callback(uint64_t id, const GoalHandleRockCentering::WrappedResult & result){
                if (id != etape) return;
                centrage_handle.reset();
                switch (result.code){
                    case rclcpp_action::ResultCode::SUCCEEDED:
                        fin_centrage(id, result.result && result.result->success,
                            result.result ? result.result->message : "résultat absent");
                        return;
                    case rclcpp_action::ResultCode::ABORTED:
                        fin_centrage(id, false, "abandonné" +
                            (result.result ? " — " + result.result->message : std::string()));
                        return;
                    case rclcpp_action::ResultCode::CANCELED:
                        fin_centrage(id, false, "annulé");
                        return;
                    default:
                        fin_centrage(id, false, "code de résultat inconnu");
                        return;
                }
            }

            // Comme dans strategie_camelot.py : un centrage raté n'est pas bloquant, on prend la pose de la carte.
            void fin_centrage(uint64_t id, bool ok, const std::string & message){
                if (id != etape) return;

                const Action centrage = actions.front();
                actions.pop_front();
                essais = 0;

                auto prise = actions.begin();
                if (!ok){
                    RCLCPP_WARN(this->get_logger(), "rock_centering : %s → pose de prise calculée sur la carte", message.c_str());
                } else {
                    RCLCPP_INFO(this->get_logger(), "rock_centering : %s", message.c_str());
                    Action corrigee;
                    if (prise != actions.end() && prise->state == State::MOVE_TO && pose_prise(centrage, corrigee)){
                        *prise = corrigee;
                    }
                }
                tick();
            }

            // Pose de prise après le centrage : on garde le cap et le latéral corrigés, on avance jusqu'à la pile.
            bool pose_prise(const Action & centrage, Action & sortie){
                geometry_msgs::msg::TransformStamped tf;
                try {
                    tf = tf_buffer->lookupTransform(REPERE, ROBOT, tf2::TimePointZero);
                } catch (const tf2::TransformException & err){
                    RCLCPP_WARN(this->get_logger(), "TF %s -> %s indisponible (%s) : pose de prise calculée sur la carte",
                                REPERE.c_str(), ROBOT.c_str(), err.what());
                    return false;
                }
                const double px = tf.transform.translation.x;
                const double py = tf.transform.translation.y;
                const auto & q = tf.transform.rotation;
                const double cap = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
                const double ux = std::cos(cap), uy = std::sin(cap);

                const double cx = bleu ? TABLE_X - centrage.x : centrage.x;
                const double d = avance_apres_centrage < 0.0
                    ? (cx - px) * ux + (centrage.y - py) * uy - DIST_PILE
                    : avance_apres_centrage;
                if (d < -0.05 || d > RECUL_CARRIERE + 0.10){
                    RCLCPP_WARN(this->get_logger(), "Avance de %.1f cm incohérente : pose de prise calculée sur la carte", d * 100.0);
                    return false;
                }

                double x = px + d * ux;
                double theta = cap * 180.0 / M_PI;
                if (bleu){                      // pose réelle : on l'écrit « côté jaune », send_move_to la remet en miroir
                    x = TABLE_X - x;
                    theta = 180.0 - theta;
                }
                sortie = {State::MOVE_TO, static_cast<float>(x), static_cast<float>(py + d * uy),
                          static_cast<float>(theta), 0, 0};
                RCLCPP_INFO(this->get_logger(), "  pose de prise corrigée : avance de %.1f cm", d * 100.0);
                return true;
            }


            // ------------------------------------------------------------ services (std_srvs/SetBool)

            void appeler_service(rclcpp::Client<SetBool>::SharedPtr client, bool valeur,
                                 const std::string & nom){
                const auto id = etape;

                if (!client->service_is_ready()){

                    echec("service " + std::string(client->get_service_name()) + " indisponible");
                    return;
                }
                auto requete = std::make_shared<SetBool::Request>();
                requete->data = valeur;

                auto futur = client->async_send_request(requete,
                    [this, id, nom](rclcpp::Client<SetBool>::SharedFuture f){ service_callback(id, nom, f); });

                const auto request_id = futur.request_id;
                lancer_watchdog(T_SERVICE, [this, client, request_id, nom](){
                    client->remove_pending_request(request_id);

                    echec(nom + " : pas de réponse à temps");
                });
            }

            void service_callback(uint64_t id, const std::string & nom, rclcpp::Client<SetBool>::SharedFuture futur){
                if (id != etape) return;
                auto reponse = futur.get();
                if (reponse->success){
                    RCLCPP_INFO(this->get_logger(), "%s : %s", nom.c_str(), reponse->message.c_str());

                    suivante();
                } else {

                    echec(nom + " : " + reponse->message);
                }
            }

};


int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Strategy>());
    if (rclcpp::ok()) rclcpp::shutdown();
    return 0;
}
