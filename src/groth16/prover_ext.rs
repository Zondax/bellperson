use std::error::Error;
use std::time::Duration;

use crate::bls::Engine;

use crate::gpu::{LockedFFTKernel, LockedMultiexpKernel};
use crate::SynthesisError;

use scheduler_client::{
    register, schedule_one_of,
    ResourceAlloc, /*Deadline, ResourceMemory, ResourceReq, ResourceType,*/
    TaskFunc, /*TaskReqBuilder, */ TaskRequirements, TaskResult,
};

pub struct FftSolver<E, F, R>
where
    for<'a> F:
        FnMut(usize, &'a mut Option<LockedFFTKernel<E>>) -> Option<Result<R, SynthesisError>>,
    E: Engine,
{
    pub accumulator: Vec<R>,
    fft_kern: Option<LockedFFTKernel<E>>,
    index: usize,
    log_d: usize,
    priority: bool,
    call: F,
    num_iter: usize,
}

impl<E, F, R> FftSolver<E, F, R>
where
    for<'a> F:
        FnMut(usize, &'a mut Option<LockedFFTKernel<E>>) -> Option<Result<R, SynthesisError>>,
    E: Engine,
{
    pub fn new(log_d: usize, priority: bool, num_iter: usize, call: F) -> Self {
        Self {
            accumulator: vec![],
            fft_kern: None,
            log_d,
            priority,
            index: 0,
            call,
            num_iter,
        }
    }

    pub fn solve(&mut self, task_req: Option<TaskRequirements>) -> Result<(), Box<dyn Error>> {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let id = rng.gen::<u32>();
        let client = match register(id, id as _) {
            Ok(c) => c,
            Err(e) => return Err(e.into()),
        };

        schedule_one_of(client, self, task_req, Duration::from_secs(90)).map_err(|e| e.into())
    }
}

impl<E, F, R> TaskFunc for FftSolver<E, F, R>
where
    for<'a> F:
        FnMut(usize, &'a mut Option<LockedFFTKernel<E>>) -> Option<Result<R, SynthesisError>>,
    E: Engine,
{
    type TaskOutput = ();

    fn init(&mut self, alloc: Option<&ResourceAlloc>) -> Result<(), Box<dyn Error>> {
        // Get the resource and create the LockedFFTKernel object
        self.fft_kern.replace(LockedFFTKernel::<E>::new(
            self.log_d,
            self.priority,
            alloc.map(|a| a.resource_id[0]),
        ));
        Ok(())
    }
    fn end(&mut self, _: Option<&ResourceAlloc>) -> Result<Self::TaskOutput, Box<dyn Error>> {
        Ok(())
    }
    fn task(&mut self, _alloc: Option<&ResourceAlloc>) -> Result<TaskResult, Box<dyn Error>> {
        if let Some(res) = (self.call)(self.index, &mut self.fft_kern) {
            match res {
                Ok(waiter) => self.accumulator.push(waiter),
                Err(e) => return Err(Box::new(e)),
            }
            self.index += 1;
            Ok(TaskResult::Continue)
        } else {
            Ok(TaskResult::Done)
        }
    }
}

pub struct MultiexpSolver<E, F, R>
where
    for<'a> F:
        FnMut(usize, &'a mut Option<LockedMultiexpKernel<E>>) -> Option<Result<R, SynthesisError>>,
    E: Engine,
{
    pub accumulator: Vec<R>,
    multiexp_kern: Option<LockedMultiexpKernel<E>>,
    index: usize,
    log_d: usize,
    priority: bool,
    call: F,
    num_iter: usize,
}

impl<E, F, R> MultiexpSolver<E, F, R>
where
    for<'a> F:
        FnMut(usize, &'a mut Option<LockedMultiexpKernel<E>>) -> Option<Result<R, SynthesisError>>,
    E: Engine,
{
    pub fn new(log_d: usize, priority: bool, num_iter: usize, call: F) -> Self {
        Self {
            accumulator: vec![],
            multiexp_kern: None,
            index: 0,
            log_d,
            priority,
            call,
            num_iter,
        }
    }

    pub fn solve(&mut self, task_req: Option<TaskRequirements>) -> Result<(), Box<dyn Error>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // get the scheduler client
        let id = rng.gen::<u32>();
        let client = match register(id, id as _) {
            Ok(c) => c,
            Err(e) => return Err(e.into()),
        };

        schedule_one_of(client, self, task_req, Duration::from_secs(90)).map_err(|e| e.into())
    }
}

impl<E, F, R> TaskFunc for MultiexpSolver<E, F, R>
where
    for<'a> F:
        FnMut(usize, &'a mut Option<LockedMultiexpKernel<E>>) -> Option<Result<R, SynthesisError>>,
    E: Engine,
{
    type TaskOutput = ();

    fn init(&mut self, alloc: Option<&ResourceAlloc>) -> Result<(), Box<dyn Error>> {
        // Get the resource and create the LockedFFTKernel object
        self.multiexp_kern.replace(LockedMultiexpKernel::<E>::new(
            self.log_d,
            self.priority,
            alloc.map(|a| a.resource_id.clone()),
        ));
        Ok(())
    }
    fn end(&mut self, _: Option<&ResourceAlloc>) -> Result<Self::TaskOutput, Box<dyn Error>> {
        Ok(())
    }
    fn task(&mut self, _alloc: Option<&ResourceAlloc>) -> Result<TaskResult, Box<dyn Error>> {
        if let Some(res) = (self.call)(self.index, &mut self.multiexp_kern) {
            match res {
                Ok(waiter) => self.accumulator.push(waiter),
                Err(e) => return Err(Box::new(e)),
            }
            self.index += 1;
            Ok(TaskResult::Continue)
        } else {
            Ok(TaskResult::Done)
        }
    }
}
