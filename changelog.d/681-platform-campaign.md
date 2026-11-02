Added
`PlatformCampaign`: runs several `RSIPlatform` cycles under the existing
`RSIController` stop policy (target, cost, patience, cycle count), cloning the
platform with a fresh seed and run id per cycle while sharing the run store,
compute, foundry and experience. An exception or incomplete cycle is reported
as an error and stops the campaign, and `PlatformCampaignReport.best_cycle`
names the highest-reward cycle (#681).
