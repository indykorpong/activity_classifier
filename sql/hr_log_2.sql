use cu_amd;

create table hr_log_2(
	event_timestamp datetime(3) not null,
     HR decimal(11,8) not null,
     user_id int,
     loaded_flag bool,
     primary key(event_timestamp, user_id, HR)
 );

insert into hr_log_2(user_id, event_timestamp, HR) select distinct user_id, event_timestamp, HR from cu_amd.hr_log;

insert into hr_log_2 (event_timestamp, HR, user_id)
select distinct event_timestamp, hr, user_id from hr_log
where event_timestamp > 
(SELECT max(event_timestamp) as max_timestamp 
FROM hr_log_2);

select * from cu_amd.hr_log_2 where loaded_flag is null;

SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));

select ID, event_timestamp, count(event_timestamp) from cu_amd.hr_log_2
group by event_timestamp having count(event_timestamp) > 1;

SET GLOBAL MAX_EXECUTION_TIME=10000;

delete t1 from cu_amd.hr_log_2 t1 inner join cu_amd.hr_log_2 t2
where t1.ID < t2.ID and t1.event_timestamp = t2.event_timestamp and t2.event_timestamp > DATE_FORMAT('2019-04-09', '%Y-%m-%d');

ALTER TABLE cu_amd.hr_log_2 AUTO_INCREMENT = 1;

-- drop table cu_amd.ActivityPeriod;
-- delete from cu_amd.hr_log_2;