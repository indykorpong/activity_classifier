create database if not exists cu_amd;
use cu_amd;

create table ActivityLog(
	Idx int not null auto_increment,
	UserID int not null, 
	DateAndTime datetime(3) not null,
	X decimal(20,8), 
	Y decimal(20,8), 
	Z decimal(20,8), 
	HR decimal(11,8),
    ActivityIndex decimal(11,8),
    Label int,
    SummarizedFlag bool,
    primary key(Idx, UserID, DateAndTime)
    );
 
create table HourlyActivitySummary(
	Idx int not null auto_increment,
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
    primary key(Idx, UserID, Date, TimeFrom)
    );
       
create table ActivityPeriod(
	Idx int not null auto_increment,
	UserID int not null,
    Date date not null,
    ActualFrom time(3) not null,
    ActualUntil time(3) not null,
    Label int,
    primary key(Idx, UserID, Date, ActualFrom)
);

create table AuditLog(
	Idx int not null auto_increment,
    StartProcessTime datetime(3) not null,
    EndProcessTime datetime(3),
    UserID int not null,
    ProcessName varchar(255) not null,
    StartData datetime(3),
    EndData datetime(3),
    ProcessStatus int not null,
    primary key(Idx, StartProcessTime, UserID, ProcessName)
);

create table UserProfile(
	Idx int not null auto_increment,
	UserID int not null,
    InitialUpdate datetime(3) not null,
    LatestUpdate datetime(3) not null,
    MinX decimal(10,8),
    MinY decimal(10,8),
    MinZ decimal(10,8),
    MaxX decimal(10,8),
    MaxY decimal(10,8),
    MaxZ decimal(10,8),
    primary key(Idx, UserID)
);