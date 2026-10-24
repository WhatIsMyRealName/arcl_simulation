#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "ros2_aruco_interfaces/msg/aruco_markers.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

class CamTracking : public rclcpp::Node{

        public:

            CamTracking() : Node("cam_tracking"){

                std::vector<double> pose_cam = declare_parameter<std::vector<double>>("pose_camera", std::vector<double>{1.275, 2.12, 1.8, -0.175, 1.045, -1.73});
                if (pose_cam.size() != 6){
                    throw std::runtime_error("pose_camera doit contenir 6 valeurs : x y z roll pitch yaw");
                }

                taille_param_  = declare_parameter<double>("taille_aruco_param", 0.07);    // marker_size donné au nœud ros2_aruco
                taille_carton_ = declare_parameter<double>("taille_aruco_carton", 0.07);
                taille_table_  = declare_parameter<double>("taille_aruco_table", 0.10);    // à vérifier dans le règlement
                frame_monde_   = declare_parameter<std::string>("frame_monde", "map");

                // monde -> capteur (convention Gazebo : x devant, y à gauche, z en haut)
                tf2::Quaternion q_cam;
                q_cam.setRPY(pose_cam[3], pose_cam[4], pose_cam[5]);
                tf2::Transform monde_capteur(q_cam, tf2::Vector3(pose_cam[0], pose_cam[1], pose_cam[2]));

                // capteur -> optique (convention OpenCV : z devant, x à droite, y en bas)
                tf2::Quaternion q_opt;
                q_opt.setRPY(-M_PI / 2, 0, -M_PI / 2);
                tf2::Transform capteur_optique(q_opt, tf2::Vector3(0, 0, 0));

                monde_optique_ = monde_capteur * capteur_optique;

                aruco_markers_sub_ = create_subscription<ros2_aruco_interfaces::msg::ArucoMarkers>(
                    "/deported_aruco_markers", 10,
                    std::bind(&CamTracking::aruco_markers_callback, this, std::placeholders::_1)
                );

                markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("deported_base_aruco_detected", 10);


                RCLCPP_INFO(this->get_logger(), "CamTracking node launched successfuly");
            }


        private:

            rclcpp::Subscription<ros2_aruco_interfaces::msg::ArucoMarkers>::SharedPtr aruco_markers_sub_;
            rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

            tf2::Transform monde_optique_;
            double taille_param_, taille_carton_, taille_table_;
            std::string frame_monde_;

            void aruco_markers_callback(const ros2_aruco_interfaces::msg::ArucoMarkers::SharedPtr markers){
                unsigned long int n = markers->marker_ids.size();
                RCLCPP_INFO(this->get_logger(), "%ld codes détécté(s)", n);

                visualization_msgs::msg::MarkerArray affichage;

                // efface les carrés du message précédent
                visualization_msgs::msg::Marker effacer;
                effacer.header.frame_id = frame_monde_;
                effacer.action = visualization_msgs::msg::Marker::DELETEALL;
                affichage.markers.push_back(effacer);

                // cartons      : 13
                // aruco_table  : 23, 22, 21, 20
                for (unsigned int i=0; i < n; i++){
                    const int64_t id = markers->marker_ids[i];

                    double taille = 0;
                    float r, g, b;
                    if (id == 13){
                        taille = taille_carton_;
                        r = 1.0; g = 0.5; b = 0.0;
                    } else if (id >= 20 && id <= 23){
                        taille = taille_table_;
                        r = 0.1; g = 0.4; b = 1.0;
                    } else{
                        continue;
                    }

                    const geometry_msgs::msg::Pose & pose = markers->poses[i];

                    double k = taille / taille_param_;
                    tf2::Transform optique_aruco(
                        tf2::Quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                        tf2::Vector3(pose.position.x * k, pose.position.y * k, pose.position.z * k)
                    );

                    tf2::Transform monde_aruco = monde_optique_ * optique_aruco;

                    tf2::Vector3 p = monde_aruco.getOrigin();
                    tf2::Matrix3x3 R = monde_aruco.getBasis();
                    tf2::Vector3 haut = R.getColumn(1);      // y du code : vers le haut du dessin
                    tf2::Vector3 normale = R.getColumn(2);   // z du code : sort de la face

                    // Sens du code (angles dans le plan de la table : 0 deg = +X, 90 deg = +Y)
                    char sens[96];
                    if (normale.z() > 0.7){
                        double cap = std::atan2(haut.y(), haut.x()) * 180.0 / M_PI;
                        snprintf(sens, sizeof(sens), "a plat, haut du code vers %.0f deg", cap);
                    }
                    else{
                        double cap = std::atan2(normale.y(), normale.x()) * 180.0 / M_PI;
                        snprintf(sens, sizeof(sens), "debout, face tournee vers %.0f deg", cap);
                    }

                    if (id == 13){
                        // Code centré sur une grande face du carton (110 mm d'épaisseur) : le centre est 5.5 cm derrière
                        tf2::Vector3 centre = p - normale * 0.055;
                    }

                    // Carré du code
                    visualization_msgs::msg::Marker carre = creer_marker(markers->header.stamp, 3*i, visualization_msgs::msg::Marker::CUBE, monde_aruco, r, g, b);
                    carre.scale.x = taille;
                    carre.scale.y = taille;
                    carre.scale.z = 0.003;
                    affichage.markers.push_back(carre);
                    
                    // ID au-dessus du code
                    tf2::Transform pose_texte(tf2::Quaternion::getIdentity(), p + normale * 0.05);
                    visualization_msgs::msg::Marker texte = creer_marker(markers->header.stamp, 3*i + 2, visualization_msgs::msg::Marker::TEXT_VIEW_FACING, pose_texte, 1.0, 1.0, 1.0);
                    texte.scale.z = 0.04;
                    texte.text = std::to_string(id);
                    affichage.markers.push_back(texte);
                }

                markers_pub_->publish(affichage);
            }

            visualization_msgs::msg::Marker creer_marker(const builtin_interfaces::msg::Time & stamp, int id, int type, const tf2::Transform & t, float r, float g, float b){
                visualization_msgs::msg::Marker m;
                m.header.frame_id = frame_monde_;
                m.header.stamp = stamp;
                m.ns = "aruco";
                m.id = id;
                m.type = type;
                m.action = visualization_msgs::msg::Marker::ADD;

                tf2::Quaternion q = t.getRotation();
                m.pose.position.x = t.getOrigin().x();
                m.pose.position.y = t.getOrigin().y();
                m.pose.position.z = t.getOrigin().z();
                m.pose.orientation.x = q.x();
                m.pose.orientation.y = q.y();
                m.pose.orientation.z = q.z();
                m.pose.orientation.w = q.w();

                m.color.r = r;
                m.color.g = g;
                m.color.b = b;
                m.color.a = 1.0;
                return m;
            }

};


int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CamTracking>());
    if (rclcpp::ok()) rclcpp::shutdown();
    return 0;
}