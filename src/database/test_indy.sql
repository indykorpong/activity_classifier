create database if not exists cu_amd;
use cu_amd;

create table ActivityLog(
	UserID int not null, 
	DateAndTime datetime(3) not null,
	X decimal(6,4), 
	Y decimal(6,4), 
	Z decimal(6,4), 
	HR decimal(7,4),
    ActivityIndex decimal(9,8),
    Label int,
    SummarizedFlag bool,
    primary key(UserID, DateAndTime)
    );
 
create table HourlyActivitySummary(
	UserID int not null,
    Date date not null,
    TimeFrom time(3) not null,
    TimeUntil time(3) not null,
    ActualFrom time(3),
    ActualUntil time(3),
    DurationSit time(3),
    DurationSleep time(3),
    DurationStand time(3),
    DurationWalk time(3),
    TotalDuration time(3),
    CountSit int,
    CountSleep int,
    CountStand int,
    CountWalk int,
    CountInactive int,
    CountActive int,
    CountTotal int,
	CountActiveToInactive int,
    DurationPerAction time(3),
    primary key(UserID, Date, TimeFrom)
    );
       
create table ActivityPeriod(
	UserID int not null,
    Date date not null,
    ActualFrom time(3) not null,
    ActualUntil time(3) not null,
    Label int,
    primary key(UserID, Date, ActualFrom)
);

create table AuditLog(
    StartTime datetime(3) not null,
    EndTime datetime(3),
    UserID int not null,
    ProcessName varchar(255) not null,
    StartingData datetime(3),
    EndingData datetime(3),
    ProcessStatus int not null,
    primary key(StartTime, UserID, ProcessName)
);

create table UserProfile(
	UserID int not null,
    MinX decimal(6,4),
    MinY decimal(6,4),
    MinZ decimal(6,4),
    MaxX decimal(6,4),
    MaxY decimal(6,4),
    MaxZ decimal(6,4),
    primary key(UserID)
);