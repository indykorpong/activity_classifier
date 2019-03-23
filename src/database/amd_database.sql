create table Patient(
	ID int not null, 
	Dateandtime datetime not null,
	X float(6,4), 
	Y float(6,4), 
	Z float(6,4), 
	HR float(6,4),
    Label varchar(255),
    primary key(ID, Dateandtime)
    );
                    
create table AllDaySummary(
	ID int not null,
    Dateandtime datetime not null,
    DurationSit time,
    DurationSleep time,
    DurationStand time,
    DurationWalk time,
    foreign key(ID, Dateandtime) references Patient(ID, Dateandtime)
    );
    
create table ActivityAnalysis(
	ID int not null,
    Dateandtime datetime not null,
    CountActive int,
    CountInactive int,
	CountTransition int,
    CountTotalActiveness int,
    DurationPerTransition time,
    foreign key(ID, Dateandtime) references Patient(ID, Dateandtime)
);
