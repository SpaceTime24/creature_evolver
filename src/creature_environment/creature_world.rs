use std::ops::{Deref, DerefMut};

use rapier3d::pipeline::PhysicsWorld;

use crate::creature_environment::creature::Creature;

#[repr(transparent)]
pub struct CreatureWorld {
    inner: Box<CreatureWorldInner<'static>>,
}

pub struct CreatureWorldInner<'a> {
    pub creature: Option<Creature<'a>>,
    pub physics: PhysicsWorld,
}

impl CreatureWorld {
    pub fn new_unpopulated<F>(mut static_environment_generator: F) -> CreatureWorld
    where
        F: FnMut(&mut PhysicsWorld),
    {
        let mut physics = PhysicsWorld::new();
        {
            static_environment_generator(&mut physics);
        }
        let inner: Box<CreatureWorldInner<'static>> = Box::new(CreatureWorldInner {
            physics,
            creature: None,
        });

        CreatureWorld { inner }
    }

    pub fn new_empty() -> CreatureWorld {
        let physics = PhysicsWorld::new();

        let inner: Box<CreatureWorldInner<'static>> = Box::new(CreatureWorldInner {
            physics,
            creature: None,
        });

        CreatureWorld { inner }
    }

    pub fn add_creature<'b, F>(&mut self, mut creature_generator: F)
    where
        F: for<'a> FnMut(&'a mut PhysicsWorld) -> Creature<'a>,
    {
        let old_creature = self.creature.take();
        drop(old_creature);
        let ptr = &raw mut self.creature;
        let ptr = (ptr as usize) as *mut Option<Creature>;
        let new_creature = creature_generator(&mut self.physics);
        unsafe {
            ptr.write(Some(new_creature));
        }
    }
}

impl Deref for CreatureWorld {
    type Target = CreatureWorldInner<'static>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for CreatureWorld {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
