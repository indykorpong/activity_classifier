create table Patient(
	ID int not null, 
	DateAndTime datetime not null,
    Milliseconds varchar(255) not null,
	X float(6,4), 
	Y float(6,4), 
	Z float(6,4), 
	HR float(7,4),
    Label varchar(255),
    primary key(ID, Dateandtime, Milliseconds)
    );

 
create table AllDaySummary(
	ID int not null,
    DateTimeFrom datetime not null,
    DateTimeUntil datetime not null,
    DurationSit time,
    DurationSleep time,
    DurationStand time,
    DurationWalk time,
    foreign key(ID) references Patient(ID)
    );
    
create table ActivityAnalysis(
	ID int not null,
    DateTimeFrom datetime not null,
    DateTimeUntil datetime not null,
    Milliseconds int not null,
    CountActive int,
    CountInactive int,
	CountTransition int,
    CountTotalActiveness int,
    DurationPerTransition time,
    foreign key(ID) references Patient(ID)
);
