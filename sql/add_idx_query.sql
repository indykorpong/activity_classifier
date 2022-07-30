insert into acc_log_2(event_timestamp, x, y, z, user_id) 
select event_timestamp, x, y, z, user_id from
(select distinct event_timestamp, x, y, z, user_id from cu_amd.accelerometer_log) as table1;

ALTER TABLE cu_amd.ActivityLog2 ADD COLUMN LoadedFlag bool default False after Label;
ALTER TABLE cu_amd.ActivityLog2 CHANGE COLUMN SummarizedFlag SummarizedFlag BOOL DEFAULT False;
ALTER TABLE cu_amd.hr_log_2 CHANGE COLUMN loaded_flag loaded_flag BOOL DEFAULT False;


ALTER TABLE cu_amd.ActivityLog ADD PRIMARY KEY (Idx);
ALTER TABLE cu_amd.ActivityLog ADD Idx INT NOT NULL AUTO_INCREMENT UNIQUE FIRST;
ALTER TABLE cu_amd.ActivityLog DROP COLUMN Idx;

ALTER TABLE cu_amd.acc_log_2 ADD INDEX (event_timestamp);
ALTER TABLE cu_amd.ActivityLog2 ADD INDEX (Z);

SELECT * FROM acc_log_2;

ALTER TABLE cu_amd.hr_log_2 ADD Idx INT NOT NULL AUTO_INCREMENT UNIQUE FIRST;

ALTER TABLE cu_amd.UserProfile CHANGE LoadedToIdx IdxToLoad int(11);

ALTER TABLE cu_amd.AuditLog ADD Idx INT NOT NULL AUTO_INCREMENT UNIQUE FIRST;