use std::fs::File;
use std::io::BufReader;
use std::thread::sleep;
use std::time::Duration;

use libjaka::JakaMini2;
use libjaka::types::{TioVout, TioVoutMode};
use robot_behavior::MotionType;
use robot_behavior::behavior::*;

fn main() -> anyhow::Result<()> {
    // let mut robot = JakaMini2::new("10.5.5.100");

    let mut robot_l = JakaMini2::new("192.168.1.2"); // 左侧机械臂
    let mut robot_r = JakaMini2::new("192.168.1.1"); // 右侧机械臂

    // 完全等同于内置函数 robot._power_on()?;
    robot_r.init()?;
    robot_r.enable()?;
    robot_l.init()?;
    robot_l.enable()?;
    robot_l.set_tio_vout(TioVout::Disable)?;
    robot_r.set_tio_vout(TioVout::Disable)?;

    // 此时你才可以发送运动指令
    robot_r.move_to(MotionType::Joint([0.; 6]))?;
    robot_l.move_to(MotionType::Joint([0.; 6]))?;
    // robot_l.move_joint(&[0.; 6]);
    // println!("{ans1:?}");
    // println!("{ans2:?}");
    // robot.move_to(MotionType::Joint([0.; 6]))?;

    robot_l.waiting_for_finish()?;
    robot_r.waiting_for_finish()?;

    let file = File::open("slave_trajectory_pick.json")?;
    let reader = BufReader::new(file);
    let path: Vec<MotionType<6>> = serde_json::from_reader(reader).unwrap();
    robot_l.move_traj_async(path)?;

    let file_2 = File::open("master_trajectory_pick.json")?;
    let reader_2 = BufReader::new(file_2);
    let path: Vec<MotionType<6>> = serde_json::from_reader(reader_2).unwrap();
    robot_r.move_traj_async(path)?;

    // robot_l.waiting_for_finish()?;
    // robot_r.waiting_for_finish()?;
    sleep(Duration::from_secs(12));
    robot_l.is_moving()?;
    robot_r.is_moving()?;

    robot_l.set_tio_vout(TioVout::Enable(TioVoutMode::V12V))?;
    robot_r.set_tio_vout(TioVout::Enable(TioVoutMode::V12V))?;

    print!("开始放置动作\n");

    let file_3 = File::open("slave_trajectory_place.json")?;
    let reader_3 = BufReader::new(file_3);
    let path: Vec<MotionType<6>> = serde_json::from_reader(reader_3).unwrap();
    robot_l.move_traj_async(path)?;

    let file_4 = File::open("master_trajectory_place.json")?;
    let reader_4 = BufReader::new(file_4);
    let path: Vec<MotionType<6>> = serde_json::from_reader(reader_4).unwrap();
    robot_r.move_traj_async(path)?;

    sleep(Duration::from_secs(15));

    robot_l.set_tio_vout(TioVout::Disable)?;
    robot_r.set_tio_vout(TioVout::Disable)?;

    // robot.disable()?;
    // 完全等同于内置函数 robot._power_off()?;
    // robot.shutdown()?;

    Ok(())
}

/// 一些测试函数可以使用快捷按钮，这使得开发一些临时使用的功能变得更加方便。你可以将其复制到任何部分，
/// 但是只有在 #[cfg(test)] 下才会被编译。
#[cfg(test)]
mod tests {
    #[test]
    fn power_on() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("10.5.5.100");
        robot._power_on()?;
        Ok(())
    }

    #[test]
    fn power_off() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("10.5.5.100");
        robot._power_off()?;
        Ok(())
    }

    #[test]
    fn enable() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("10.5.5.100");
        robot._enable()?;
        Ok(())
    }

    #[test]
    fn disable() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("10.5.5.100");
        robot._disable()?;
        Ok(())
    }
}
