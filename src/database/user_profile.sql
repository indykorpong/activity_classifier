SELECT * FROM cu_amd.UserProfile;

delete from UserProfile where UserID=17;

SELECT @UserID:=17;
select @UserID;
SELECT @LastResultIdx := Idx FROM ActivityLog WHERE (LoadedFlag=False or Label IS NULL) and UserID=@UserID ORDER BY Idx DESC LIMIT 1;
SELECT @LastResultIdx := Idx FROM ActivityLog WHERE UserID=@UserID ORDER BY Idx DESC LIMIT 1;

SELECT @IdxToUpdate := Idx+1 FROM (SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog WHERE UserID=@UserID and Idx<=@LastResultIdx and Idx<@IdxToLoad_1+5000 and Label IS NOT NULL order by Idx ASC) as table2;
UPDATE cu_amd.UserProfile set IdxToLoad = 25001 where UserID=17;

UPDATE cu_amd.UserProfile set MinX=null where UserID=17;
UPDATE cu_amd.UserProfile set MinY=null where UserID=17;
UPDATE cu_amd.UserProfile set MinZ=null where UserID=17;

UPDATE cu_amd.UserProfile set MaxX=null where UserID=17;
UPDATE cu_amd.UserProfile set MaxY=null where UserID=17;
UPDATE cu_amd.UserProfile set MaxZ=null where UserID=17;

SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog2 WHERE UserID=@UserID and Idx<=@LastResultIdx and Label IS NOT NULL order by Idx DESC limit 1;
SELECT @IdxToUpdate := Idx+1 FROM (SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog WHERE UserID=@UserID and Idx<=@LastResultIdx and Label IS NOT NULL order by Idx ASC limit 1) as table2;
SELECT IdxToLoad FROM UserProfile WHERE UserID=@UserID;

select * from cu_amd.ActivityLog where UserID=10 and Idx>=2324404;

SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog2 WHERE Label IS NULL and UserID=@UserID and Idx<=@LastResultIdx and Idx>=@IdxToLoad_1 and Idx<@IdxToLoad_1+5000;
SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog2 WHERE Label IS NULL and UserID=@UserID;

SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog2 WHERE UserID=@UserID and Idx<=@LastResultIdx and Label IS NOT NULL order by Idx ASC limit 1