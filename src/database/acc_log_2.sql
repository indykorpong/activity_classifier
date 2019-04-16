use cu_amd;

create table acc_log_2(
	ID int not null,
    x decimal(9,8),
    y decimal(9,8),
    z decimal(9,8),
    event_timestamp datetime(3) not null,
    create_timestamp datetime(3),
    user_id int,
    primary key(event_timestamp, ID)
);

INSERT acc_log_2 SELECT * FROM accelerometer_log;

select * from cu_amd.acc_log_2;

SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));

select ID, event_timestamp, count(event_timestamp) from cu_amd.acc_log_2
group by event_timestamp having count(event_timestamp) > 1;

delete t1 from cu_amd.acc_log_2 t1 inner join cu_amd.acc_log_2 t2
where t1.ID < t2.ID and t1.event_timestamp = t2.event_timestamp;

ALTER TABLE cu_amd.acc_log_2 AUTO_INCREMENT = 1;

-- drop table cu_amd.acc_log_2;