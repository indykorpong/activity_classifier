SELECT * FROM cu_amd.acc_log_2 WHERE user_id='17' and event_timestamp>DATE_FORMAT("2019-04-29", "%Y-%m-%d");
SELECT * FROM cu_amd.ActivityLog2 WHERE UserID='17' and DateAndTime>DATE_FORMAT("2019-05-01", "%Y-%m-%d");

ALTER TABLE cu_amd.acc_log_2 MODIFY COLUMN loaded_flag bool NOT NULL DEFAULT 0;

delete from cu_amd.ActivityPeriod;

-- remove rows whose date > 30-04-2019
update cu_amd.acc_log_2 set loaded_flag=False where event_timestamp>DATE_FORMAT("2019-04-29 12:00:00", "%Y-%m-%d %H-%i-%S"); -- and event_timestamp<DATE_FORMAT("2019-05-01 21:30:00", "%Y-%m-%d %H-%i-%S");
update cu_amd.hr_log_2 set loaded_flag=False where event_timestamp>DATE_FORMAT("2019-04-29 12:00:00", "%Y-%m-%d %H-%i-%S"); -- and event_timestamp<DATE_FORMAT("2019-05-01 21:01:00", "%Y-%m-%d %H-%i-%S");

delete from cu_amd.ActivityLog2 where DateAndTime>DATE_FORMAT("2019-05-10 12:00:00", "%Y-%m-%d %H-%i-%S");
update cu_amd.ActivityLog2 set LoadedFlag=False where UserID=17 and Idx>=0 and Idx<35001;
update cu_amd.ActivityLog2 set SummarizedFlag=False where UserID=17 and Idx>=0 and Idx<35001;

ALTER TABLE ActivityLog2 AUTO_INCREMENT = 1;
delete from cu_amd.HourlyActivitySummary;
ALTER TABLE HourlyActivitySummary AUTO_INCREMENT = 1;

delete from cu_amd.ActivityLog2;
ALTER TABLE ActivityLog2 AUTO_INCREMENT = 1;

update cu_amd.ActivityLog2 set LoadedFlag=False where UserID=17 and Idx>=0 and Idx<35001;
update cu_amd.ActivityLog2 set SummarizedFlag=False where UserID=17 and Idx>=0 and Idx<35001;
delete from cu_amd.ActivityLog where X is null or Y is null or Z is null or HR is null;

insert into ActivityLog2(UserID, DateAndTime, X, Y, Z, Label) 
select distinct UserID, DateAndTime, X, Y, Z, Label from ActivityLog;

-- delete from cu_amd.ActivityLog where UserID=17 and (X is null or Y is null or Z is null);

SELECT * FROM cu_amd.UserProfile;

ALTER TABLE cu_amd.UserProfile ADD COLUMN IdxToLoad int AFTER UserID;
ALTER TABLE cu_amd.UserProfile ALTER IdxToLoad SET DEFAULT 0;

-- load batch by batch
Select @UserID := 17;
SELECT @IdxToLoad1 := Idx from (
	SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog2 WHERE Label IS NULL and UserID=@UserID
) AS table1 limit 1;

SELECT @IdxToLoad2 := IdxToLoad from (
	SELECT IdxToLoad FROM UserProfile WHERE UserID=@UserID
) AS table1 limit 1;

SELECT @IdxToLoad := IF(@IdxToLoad2 > @IdxToLoad1, @IdxToLoad2, @IdxToLoad1);
SELECT @IdxToLoad_1 := IF(@IdxToLoad IS NULL, 0, @IdxToLoad);

-- loop starts
SELECT @LastResultIdx := Idx FROM ActivityLog2 WHERE Label IS NULL and UserID=@UserID ORDER BY Idx DESC LIMIT 1;
SELECT @FirstResultIdx := Idx FROM ActivityLog2 WHERE Label IS NULL and UserID=@UserID ORDER BY Idx ASC LIMIT 1;
SELECT count(*) FROM ActivityLog2 WHERE Label IS NULL and X IS NOT NULL and UserID=@UserID ORDER BY Idx DESC LIMIT 1;
-- get data
-- '636594'
SELECT @LastResultIdx, @IdxToLoad_1, @IdxToLoad_1 + 5000, @UserID;
SELECT * FROM ActivityLog2 WHERE UserID=17 and Idx<=@LastResultIdx and Idx>=@IdxToLoad_1 and Idx<@IdxToLoad_1+5000 and Label IS NULL;

SELECT @IdxLoaded := IF(@IdxToLoad_0+5000>@LastResultIdx, @LastResultIdx, @IdxToLoad_1+5000);

SELECT @IdxToUpdate := Idx FROM (SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog2 WHERE UserID=@UserID and Idx<=@LastResultIdx and Label IS NULL order by Idx asc limit 1) as table2;
select @IdxToUpdate;
-- update idx to load and save it to user profile table
update cu_amd.UserProfile set IdxToLoad = @IdxToLoad + 1000 where UserID=@UserID;


-- loop ends