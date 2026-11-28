"""Membership checks for a cumulative update, independent of the tensor runtime."""


def validate_cumulative_membership(new_rows, replay_rows, parent_membership):
    def mapping(rows):
        by_id = {row['sample_id']: row['pixel_sha256'] for row in rows}
        if len(by_id) != len(rows) or len(set(by_id.values())) != len(rows):
            raise ValueError('duplicate cumulative membership')
        return by_id

    parent = mapping(parent_membership)
    replay = mapping(replay_rows)
    new = mapping(new_rows)
    if not parent or not new:
        raise ValueError('cumulative update requires a nonempty parent and new bank')
    if parent != replay:
        raise ValueError('cumulative replay must preserve every exact parent training member')
    if set(new) & set(parent) or set(new.values()) & set(parent.values()):
        raise ValueError('new bank contains a previously exposed parent frame')
    for row in replay_rows:
        if row.get('source_role') not in ['f0_train', 'previously_admitted']:
            raise ValueError('unverified cumulative replay source role')
    return dict(parent_training_rows=len(parent), new_unique_rows=len(new),
                cumulative_training_rows=len(parent)+len(new), parent_membership_preserved=True)
