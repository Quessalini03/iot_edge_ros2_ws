#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "iot_interfaces/msg/temperature_humidity.hpp"

using namespace std::chrono_literals;

class FirePublisher : public rclcpp::Node
{
public:
  FirePublisher()
  : Node("fire_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<iot_interfaces::msg::TemperatureHumidity>("temperature_humidity_data", 10);
    timer_ = this->create_wall_timer(1000ms, std::bind(&FirePublisher::timer_callback, this));
    std::srand(std::time(nullptr));
    base_temp_ = 22.0f + static_cast<float>(std::rand() % 500 - 250) / 100.0f;
    base_hum_ = 50.0f + static_cast<float>(std::rand() % 100 - 50) / 10.0f;
  }

private:
  void timer_callback()
  {
    float temp = base_temp_;
    float hum = base_hum_;
    if (count_ >= 2) {
      temp += 10.0f * (count_ - 1);  // Rapid temp rise
      hum -= 10.0f * (count_ - 1);   // Rapid humidity drop
    }

    // Add noise
    temp += static_cast<float>(std::rand() % 400 - 200) / 100.0f;
    hum += static_cast<float>(std::rand() % 600 - 300) / 100.0f;

    temp = std::clamp(temp, -40.0f, 80.0f);
    hum = std::clamp(hum, 0.0f, 100.0f);

    iot_interfaces::msg::TemperatureHumidity msg;
    msg.temperature = temp;
    msg.humidity = hum;

    RCLCPP_INFO(this->get_logger(), "FIRE Seq - Temp: %.2f, Hum: %.2f", temp, hum);
    publisher_->publish(msg);

    count_ = (count_ + 1) % 5;
    if (count_ == 0) {
      base_temp_ = 22.0f + static_cast<float>(std::rand() % 500 - 250) / 100.0f;
      base_hum_ = 50.0f + static_cast<float>(std::rand() % 100 - 50) / 10.0f;
    }
  }

  float base_temp_;
  float base_hum_;
  size_t count_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<iot_interfaces::msg::TemperatureHumidity>::SharedPtr publisher_;
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FirePublisher>());
  rclcpp::shutdown();
  return 0;
}