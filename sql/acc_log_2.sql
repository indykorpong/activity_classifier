use cu_amd;

SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));

SET GLOBAL MAX_EXECUTION_TIME=1000;

create table acc_log_2(
	event_timestamp datetime(3) not null,
     x decimal(20,8),
     y decimal(20,8),
     z decimal(20,8),
     user_id int,
     loaded_flag bool,
     primary key(event_timestamp, user_id)
 );

SELECT * FROM acc_log_2;

delete from acc_log_2;

insert into acc_log_2(event_timestamp, x, y, z, user_id) 
select event_timestamp, x, y, z, user_id from
(select distinct event_timestamp, x, y, z, user_id from cu_amd.accelerometer_log) as table1;

SELECT * FROM cu_amd.accelerometer_log;

delete from ActivityLog
where DateAndTime > date_format('2019-05-01 00:00:00.000', '%Y-%m-%d %H:%i:%S.%f') 
or UserID = 18;

delete from acc_log_2
where event_timestamp > date_format('2019-05-01 00:00:00.000', '%Y-%m-%d %H:%i:%S.%f')
or user_id = 18;

delete from hr_log_2
where event_timestamp > date_format('2019-05-01 00:00:00.000', '%Y-%m-%d %H:%i:%S.%f')
or user_id = 18;

delete from ActivityPeriod
where Date >= date_format('2019-05-01', '%Y-%m-%d')
or UserID = 18;

insert into cu_amd.acc_log_2 (event_timestamp, x, y, z, user_id)
select distinct event_timestamp, x, y, z, user_id from cu_amd.accelerometer_log 
where event_timestamp > 
(SELECT max(event_timestamp) as max_timestamp 
FROM cu_amd.acc_log_2);

SELECT `AUTO_INCREMENT`
FROM  INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'cu_amd'
AND TABLE_NAME = 'accelerometer_log';

delete from acc_log_2;