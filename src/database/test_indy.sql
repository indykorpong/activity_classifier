create database if not exists cu_amd;
use cu_amd;

create table Patient(
	ID int not null, 
	DateAndTime datetime(3) not null,
	X decimal(6,4), 
	Y decimal(6,4), 
	Z decimal(6,4), 
	HR decimal(7,4),
    ActivityIndex decimal(9,8),
    Label int,
    primary key(ID, DateAndTime)
    );
 
create table AllDaySummary(
	ID int not null,
    Date date not null,
    TimeFrom time not null,
    TimeUntil time not null,
    ActualFrom time not null,
    ActualUntil time not null,
    DurationSit time,
    DurationSleep time,
    DurationStand time,
    DurationWalk time,
    TotalDuration time,
    CountSit int,
    CountSleep int,
    CountStand int,
    CountWalk int,
    CountInactive int,
    CountActive int,
    CountTotalActiveness int,
	CountTransition int,
    DurationPerTransition time
    );
       
create table ActivityPeriod(
	ID int not null,
    Date date not null,
    TimeFrom time not null,
    TimeUntil time not null,
    Label int
);

create table Logging(
    StartTime datetime(3) not null,
    StopTime datetime(3),
    ProcessName varchar(255) not null,
    ProcessStatus int not null,
    primary key(StartTime, ProcessName)
);