use cu_amd;

create table acc_log_2(
	event_timestamp datetime(3) not null,
     x decimal(9,8),
     y decimal(9,8),
     z decimal(9,8),
     user_id int,
     loaded_flag bool,
     primary key(event_timestamp, user_id)
 );

SELECT * FROM acc_log_2;

insert into acc_log_2(event_timestamp, x, y, z, user_id) select distinct event_timestamp, x, y, z, user_id from cu_amd.accelerometer_log;

SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));

select ID, event_timestamp, count(event_timestamp) from cu_amd.acc_log_2
group by event_timestamp having count(event_timestamp) > 1;

SET GLOBAL MAX_EXECUTION_TIME=1000;

delete t1 from cu_amd.acc_log_2 t1, cu_amd.acc_log_2 t2
where t1.ID < t2.ID and t1.event_timestamp = t2.event_timestamp;

ALTER TABLE cu_amd.acc_log_2 AUTO_INCREMENT = 1;

-- drop table cu_amd.acc_log_2;

SELECT * FROM acc_log_2;
update cu_amd.acc_log_2 set loaded_flag=NULL;