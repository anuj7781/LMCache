// SPDX-License-Identifier: Apache-2.0

use std::io;

use super::{
    check_nvme_ioctl_result, is_transient_submit_error, placement_id_to_u16, SqeTracker,
    SubmitOutcome,
};

#[test]
fn check_nvme_ioctl_result_accepts_success() {
    assert!(check_nvme_ioctl_result(0, "NVMe ioctl failed").is_ok());
}

#[test]
fn check_nvme_ioctl_result_rejects_nvme_status() {
    assert!(check_nvme_ioctl_result(1, "NVMe ioctl failed").is_err());
}

#[test]
fn placement_id_to_u16_accepts_valid_bounds() {
    assert_eq!(placement_id_to_u16(1).unwrap(), 1);
    assert_eq!(placement_id_to_u16(65535).unwrap(), 65535);
}

#[test]
fn placement_id_to_u16_rejects_reserved_and_out_of_range_values() {
    assert!(placement_id_to_u16(0).is_err());
    assert!(placement_id_to_u16(-1).is_err());
    assert!(placement_id_to_u16(65536).is_err());
}

fn tracker_with(user_data: &[u64]) -> SqeTracker {
    let mut tracker = SqeTracker::default();
    for &ud in user_data {
        tracker.pushed(ud);
    }
    tracker
}

#[test]
fn submit_errors_classify_transient_vs_fatal() {
    assert!(is_transient_submit_error(Some(libc::EAGAIN)));
    assert!(is_transient_submit_error(Some(libc::EBUSY)));
    assert!(is_transient_submit_error(Some(libc::EINTR)));
    assert!(!is_transient_submit_error(Some(libc::EBADF)));
    assert!(!is_transient_submit_error(Some(libc::EINVAL)));
    assert!(!is_transient_submit_error(None));
}

#[test]
fn sqe_tracker_partial_submit_keeps_unconsumed_tail() {
    let mut tracker = tracker_with(&[1, 2, 3]);
    assert_eq!(tracker.on_submit(&Ok(2)), SubmitOutcome::Progress);
    // Only the unconsumed SQE is reported if a later submit fails fatally.
    let fatal = tracker.on_submit(&Err(io::Error::from_raw_os_error(libc::EBADF)));
    assert_eq!(fatal, SubmitOutcome::Fatal(vec![3]));
}

#[test]
fn sqe_tracker_transient_error_retains_all_pending() {
    let mut tracker = tracker_with(&[4, 5]);
    let retry = tracker.on_submit(&Err(io::Error::from_raw_os_error(libc::EAGAIN)));
    assert_eq!(retry, SubmitOutcome::Retry);
    assert_eq!(tracker.on_submit(&Ok(2)), SubmitOutcome::Progress);
    // Everything was consumed, so nothing is left to fail.
    let fatal = tracker.on_submit(&Err(io::Error::from_raw_os_error(libc::EBADF)));
    assert_eq!(fatal, SubmitOutcome::Fatal(vec![]));
}

#[test]
fn sqe_tracker_fatal_error_reports_each_unsubmitted_request_once() {
    let mut tracker = tracker_with(&[7, 8, 9]);
    let fatal = tracker.on_submit(&Err(io::Error::from_raw_os_error(libc::EINVAL)));
    assert_eq!(fatal, SubmitOutcome::Fatal(vec![7, 8, 9]));
    let again = tracker.on_submit(&Err(io::Error::from_raw_os_error(libc::EINVAL)));
    assert_eq!(again, SubmitOutcome::Fatal(vec![]));
}

#[test]
fn sqe_tracker_over_reported_consumption_is_clamped() {
    let mut tracker = tracker_with(&[10]);
    assert_eq!(tracker.on_submit(&Ok(5)), SubmitOutcome::Progress);
    tracker.pushed(11);
    let fatal = tracker.on_submit(&Err(io::Error::from_raw_os_error(libc::EBADF)));
    assert_eq!(fatal, SubmitOutcome::Fatal(vec![11]));
}
