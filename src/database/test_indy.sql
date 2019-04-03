create database if not exists test_indy;
use test_indy;

create table Patient(
	ID int not null, 
	DateAndTime datetime(3) not null,
	X float(6,4), 
	Y float(6,4), 
	Z float(6,4), 
	HR float(7,4),
    Label int,
    primary key(ID, Dateandtime)
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
    CountActive int,
    CountInactive int,
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