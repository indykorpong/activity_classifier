use cu_amd;

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

select distinct event_timestamp, x, y, z, user_id from cu_amd.accelerometer_log as table1;

insert into acc_log_2(event_timestamp, x, y, z, user_id) 
select event_timestamp, x, y, z, user_id from
(select distinct event_timestamp, x, y, z, user_id from cu_amd.accelerometer_log) as table1;

SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));

select ID, event_timestamp, count(event_timestamp) from cu_amd.acc_log_2
group by event_timestamp having count(event_timestamp) > 1;

SET GLOBAL MAX_EXECUTION_TIME=1000;

delete t1 from cu_amd.acc_log_2 t1, cu_amd.acc_log_2 t2
where t1.ID < t2.ID and t1.event_timestamp = t2.event_timestamp;

insert into acc_log_2 (event_timestamp, x, y, z, user_id)
select distinct event_timestamp, x, y, z, user_id from accelerometer_log
where event_timestamp > 
(SELECT max(event_timestamp) as max_timestamp 
FROM acc_log_2);

SELECT * FROM cu_amd.accelerometer_log;

SELECT * FROM acc_log_2;
update cu_amd.acc_log_2 set loaded_flag=NULL;

delete from ActivityLog
where DateAndTime > date_format('2019-04-30 16:48:56.787', '%Y-%m-%d %H:%i:%S.%f') 
or UserID = 18;

delete from acc_log_2
where event_timestamp > date_format('2019-04-30 16:48:56.787', '%Y-%m-%d %H:%i:%S.%f')
or user_id = 18;

delete from ActivityPeriod
where Date = date_format('2019-04-30', '%Y-%m-%d')
and ActualFrom >= time_format('16:48:56.787', '%H:%i:%S.%f')
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

create table MaxTimestamp(
	UserID int not null,
	MaxTimestamp datetime(3) not null,
    primary key(UserID, MaxTimestamp)
);

delete from acc_log_2;