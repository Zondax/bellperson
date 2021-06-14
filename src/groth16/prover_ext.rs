use super::BellTaskType;
use crate::bls::Engine;

use crate::gpu::{LockedFFTKernel, LockedMultiexpKernel};
use crate::SynthesisError;
#[cfg(feature = "gpu")]
use log::warn;

#[cfg(feature = "gpu")]
use scheduler_client::{
    execute_without_scheduler, register, resources_as_requirements, schedule_one_of,
    Error as ClientError, ResourceAlloc, ResourceMemory, ResourceType, TaskFunc, TaskReqBuilder,
    TaskResult, TaskType,
};

// this timeout represents the amount of time a task can wait either for resources or preemption
// after which and error(ClientError::Timeout) will be returned to the caller.
// ideally different tasks would have a different timeouts. for example, for the case of winning
// post we want a timeout that allows us to wait a reasonble amount of time for resources and that
// gives us enough time to fallback to cpu in case of an error.
// later we can discuss if this value stays here as an argument or in a configuration file.
#[cfg(feature = "gpu")]
const TIMEOUT: u64 = 120;

macro_rules! solver {
    ($class:ident, $kern:ident) => {
        pub struct $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
            E: Engine,
        {
            pub accumulator: Vec<R>,
            kernel: Option<$kern<E>>,
            index: usize,
            _log_d: usize,
            call: F,
        }

        impl<E, F, R> $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
            E: Engine,
        {
            pub fn new(log_d: usize, call: F) -> Self {
                $class::<E, F, R> {
                    accumulator: vec![],
                    kernel: None,
                    index: 0,
                    _log_d: log_d,
                    call,
                }
            }

            #[cfg(feature = "gpu")]
            pub fn solve(&mut self, task_type: Option<BellTaskType>) -> Result<(), SynthesisError> {
                use rand::Rng;
                use std::time::Duration;

                let mut rng = rand::thread_rng();
                // use a random number as client id.
                let id = rng.gen::<u32>();
                let client = register::<SynthesisError>(id, id as _)?;
                let task_type = task_type.map(|t| match t {
                    BellTaskType::WinningPost => TaskType::WinningPost,
                    BellTaskType::WindowPost => TaskType::WindowPost,
                    _ => TaskType::MerkleProof,
                });
                //let mut fft_requirements = requirements.clone();
                //// Fft runs only on 1 GPU
                //fft_requirements = fft_requirements.and_then(|mut r| {
                //r.req.iter_mut().for_each(|req| req.quantity = 1);
                //Some(r)
                //});
                // Retrieves the current resources on the system
                // this also includes the ones that are in use.
                // and construct a ResourceReq from it. Ideally
                // we should have an average on how much memory
                // this would take in order to optimize the GPUs memory
                // usage. This is just a helper function and might be sweep out or modified.
                let resource_req = resources_as_requirements()?;
                let mut task_req = TaskReqBuilder::new();
                if let Some(task_type) = task_type {
                    task_req = task_req.with_task_type(task_type);
                }

                for mut req in resource_req.into_iter() {
                    req.resource = ResourceType::Gpu(ResourceMemory::All);
                    task_req = task_req.resource_req(req);
                }
                let requirements = task_req.build();

                let task_type = requirements.task_type;
                let mut result =
                    schedule_one_of(client, self, requirements, Duration::from_secs(TIMEOUT));

                // handle the case where gpu fails or the resource did not get any resource
                // from the scheduler. winning-post onl
                if task_type == Some(TaskType::WinningPost) {
                    if let Err(SynthesisError::Scheduler(ClientError::Timeout)) = result {
                        warn!("Timeout error, switching back to CPU for task WinningPost");
                        result = execute_without_scheduler(self);
                    }
                }
                result
            }

            #[cfg(not(feature = "gpu"))]
            pub fn solve(&mut self, _: Option<BellTaskType>) -> Result<(), SynthesisError> {
                while let Some(res) = (self.call)(self.index, &mut self.kernel) {
                    match res {
                        Ok(res) => self.accumulator.push(res),
                        Err(e) => return Err(e),
                    }
                    self.index += 1;
                }
                Ok(())
            }
        }

        #[cfg(feature = "gpu")]
        impl<E, F, R> TaskFunc for $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
            E: Engine,
        {
            type Output = ();
            type Error = SynthesisError;

            fn init(&mut self, alloc: Option<&ResourceAlloc>) -> Result<Self::Output, Self::Error> {
                self.kernel.replace($kern::<E>::new(self._log_d, alloc));
                Ok(())
            }
            fn end(&mut self, _: Option<&ResourceAlloc>) -> Result<Self::Output, Self::Error> {
                Ok(())
            }
            fn task(&mut self, _alloc: Option<&ResourceAlloc>) -> Result<TaskResult, Self::Error> {
                if let Some(res) = (self.call)(self.index, &mut self.kernel) {
                    match res {
                        Ok(res) => self.accumulator.push(res),
                        Err(e) => return Err(e),
                    }
                    self.index += 1;
                    Ok(TaskResult::Continue)
                } else {
                    Ok(TaskResult::Done)
                }
            }
        }
    };
}

solver!(FftSolver, LockedFFTKernel);
solver!(MultiexpSolver, LockedMultiexpKernel);
