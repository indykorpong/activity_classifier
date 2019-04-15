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

select ID, event_timestamp, count(event_timestamp) from cu_amd.acc_log_2
having count(event_timestamp) > 1;

-- drop table cu_amd.acc_log_2;