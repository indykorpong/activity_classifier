SELECT * FROM cu_amd.HourlyActivitySummary;

delete from cu_amd.HourlyActivitySummary;

select DurationSit from cu_amd.HourlyActivitySummary where TimeFrom='00:00:00.00';

set @OldDurationSit := (select DurationSit from cu_amd.HourlyActivitySummary where TimeFrom='00:00:00.00');