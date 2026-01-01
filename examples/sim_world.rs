use robot_behavior::MotionType;
use std::fs::File;
use std::io::BufReader;

use nalgebra::{self as na, Vector3};
use std::{f64::consts::FRAC_PI_2, time::Duration};

use libjaka::JakaMini2;
use robot_behavior::{Entity, behavior::*};
use rsbullet::{RsBullet, RsBulletRobot};

fn main() -> anyhow::Result<()> {
    let mut physics_engine = RsBullet::new(rsbullet::Mode::Gui)?;

    // TODO : change the path to your own
    physics_engine
        .add_search_path("E:\\jaka\\mini_exam\\asserts")?
        .set_gravity([0., 0., -10.])?
        .set_step_time(Duration::from_secs_f64(1. / 125.))?;

    let mut robot_1: RsBulletRobot<JakaMini2> = physics_engine
        .robot_builder::<JakaMini2>("robot_1")
        .base(na::Isometry3::from_parts(
            [0.0, 0.275, 0.0].into(),
            na::Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2).into(),
        ))
        .scaling(1.0)
        .base_fixed(true)
        .load()?;
    let mut robot_2 = physics_engine
        .robot_builder::<JakaMini2>("robot_2")
        .base(na::Isometry3::from_parts(
            [0.0, -0.275, 0.0].into(),
            na::Rotation3::from_axis_angle(&Vector3::z_axis(), -FRAC_PI_2).into(),
        ))
        .base_fixed(true)
        .load()?;

    physics_engine
        .visual(Entity::Box {
            half_extents: [0.01, 0.375, 0.01],
        })
        .base([0.375, 0., 0.255])
        .load()?;
    physics_engine
        .visual(Entity::Box {
            half_extents: [0.01, 0.375, 0.01],
        })
        .base([0.375, 0., 0.26])
        .load()?;
    physics_engine
        .visual(Entity::Box {
            half_extents: [0.01, 0.01, 0.3],
        })
        .base([0.2125, 0.1875, 0.15])
        .load()?;
    physics_engine
        .visual(Entity::Box {
            half_extents: [0.01, 0.01, 0.3],
        })
        .base([0.2125, -0.1875, 0.15])
        .load()?;
    physics_engine
        .visual(Entity::Box {
            half_extents: [0.5, 0.5, 0.5],
        })
        .base([0.1, 0., -0.25])
        .load()?;

    for _ in 0..100 {
        physics_engine.step()?;
    }
    robot_1.move_joint(&[0.; 6])?;
    robot_2.move_joint(&[0.; 6])?;
    // robot_2.move_joint(&[FRAC_PI_2; 6])?;

    // robot_1.move_traj_from_file("output_traj_zero.json")?;
    // robot_2.move_traj_from_file("robot_2_traj.json")?;

    let file = File::open("master_trajectory_full.json")?;
    let reader = BufReader::new(file);
    let path: Vec<MotionType<6>> = serde_json::from_reader(reader).unwrap();
    robot_2.move_traj_async(path)?;

    let file_2 = File::open("slave_trajectory_full.json")?;
    let reader_2 = BufReader::new(file_2);
    let path: Vec<MotionType<6>> = serde_json::from_reader(reader_2).unwrap();
    robot_1.move_traj_async(path)?;

    loop {
        std::thread::sleep(Duration::from_millis(10));
        physics_engine.step()?;
    }
}
