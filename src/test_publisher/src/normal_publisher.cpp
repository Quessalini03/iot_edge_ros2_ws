#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "iot_interfaces/msg/temperature_humidity.hpp"

using namespace std::chrono_literals;

class NormalPublisher : public rclcpp::Node
{
public:
  NormalPublisher()
  : Node("normal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<iot_interfaces::msg::TemperatureHumidity>("temperature_humidity_data", 10);
    timer_ = this->create_wall_timer(1000ms, std::bind(&NormalPublisher::timer_callback, this));
    std::srand(std::time(nullptr));
  }

private:
  void timer_callback()
  {
    float temp = std::max(-40.0f, std::min(80.0f, 22.0f + static_cast<float>(std::rand() % 600 - 300) / 100.0f));
    float hum = std::max(0.0f, std::min(100.0f, 50.0f + static_cast<float>(std::rand() % 200 - 100) / 10.0f));

    iot_interfaces::msg::TemperatureHumidity msg;
    msg.temperature = temp;
    msg.humidity = hum;

    RCLCPP_INFO(this->get_logger(), "Normal Seq - Temp: %.2f, Hum: %.2f", temp, hum);
    publisher_->publish(msg);
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<iot_interfaces::msg::TemperatureHumidity>::SharedPtr publisher_;
  size_t count_;
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NormalPublisher>());
  rclcpp::shutdown();
  return 0;
}