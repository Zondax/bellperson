use std::error::Error;
use std::time::Duration;

use crate::bls::Engine;

use crate::gpu::{LockedFFTKernel, LockedMultiexpKernel};
use crate::SynthesisError;

use scheduler_client::{
    register, schedule_one_of, Deadline, ResourceAlloc, ResourceMemory, ResourceReq, ResourceType,
    TaskFunc, TaskReqBuilder, TaskResult,
};

pub(crate) struct FftSolver<E, F, R>
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

    pub fn solve(&mut self) -> Result<(), Box<dyn Error>> {
        // get the scheduler client
        let client = match register(std::process::id() as _, std::process::id() as _) {
            Ok(c) => c,
            Err(e) => return Err(e.into()),
        };

        let requirements = if cfg!(feature = "gpu") {
            let start = chrono::Utc::now();
            let end = start + chrono::Duration::seconds(30);
            let deadline = Deadline::new(start, end);

            match TaskReqBuilder::new()
                .resource_req(ResourceReq {
                    resource: ResourceType::Gpu(ResourceMemory::Mem(2)),
                    quantity: 1,
                    preemptible: true,
                })
                .with_time_estimations(
                    Duration::from_millis(500),
                    self.num_iter,
                    Duration::from_millis(3000),
                )
                .with_deadline(deadline)
                .build()
            {
                Ok(req) => Some(req),
                Err(e) => return Err(e.into()),
            }
        } else {
            None
        };
        schedule_one_of(client, self, requirements, Duration::from_secs(10)).map_err(|e| e.into())
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

pub(crate) struct MultiexpSolver<E, F, R>
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

    pub fn solve(&mut self) -> Result<(), Box<dyn Error>> {
        // get the scheduler client
        let client = match register(std::process::id() as _, std::process::id() as _) {
            Ok(c) => c,
            Err(e) => return Err(e.into()),
        };

        let requirements = if cfg!(feature = "gpu") {
            let start = chrono::Utc::now();
            let end = start + chrono::Duration::seconds(30);
            let deadline = Deadline::new(start, end);

            match TaskReqBuilder::new()
                .resource_req(ResourceReq {
                    resource: ResourceType::Gpu(ResourceMemory::Mem(2)),
                    quantity: 1,
                    preemptible: true,
                })
                .with_time_estimations(
                    Duration::from_millis(500),
                    self.num_iter,
                    Duration::from_millis(3000),
                )
                .with_deadline(deadline)
                .build()
            {
                Ok(req) => Some(req),
                Err(e) => return Err(e.into()),
            }
        } else {
            None
        };
        schedule_one_of(client, self, requirements, Duration::from_secs(10)).map_err(|e| e.into())
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
