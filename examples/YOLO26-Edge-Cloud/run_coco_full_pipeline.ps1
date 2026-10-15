param(
    [string]$PythonExe = "python",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$DataRoot = "F:\IDE\datasets\coco",
    [string]$Device = "0",
    [string]$RunTag = "coco_full_v1",
    [switch]$SkipTrain,
    [switch]$SkipRuleBaseline,
    [switch]$SkipSummary
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$Args
    )
    Write-Host ""
    Write-Host "==== $Title ====" -ForegroundColor Cyan
    Write-Host "$PythonExe $($Args -join ' ')" -ForegroundColor DarkGray
    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title"
    }
}

Set-Location $RepoRoot

$TrainList = Join-Path $DataRoot "train2017.txt"
$ValList = Join-Path $DataRoot "val2017.txt"
$PolicyDir = "runs\edge_cloud\learned_policy_$RunTag"
$PolicyPath = Join-Path $RepoRoot "$PolicyDir\learned_upload_policy.json"

$RuleDir = "runs\edge_cloud\eval_${RunTag}_rule_aggressive"
$Learned70Dir = "runs\edge_cloud\eval_${RunTag}_learned_thr070_cloud512"
$Learned60Dir = "runs\edge_cloud\eval_${RunTag}_learned_thr060_cloud512"
$AdaptiveDir = "runs\edge_cloud\eval_${RunTag}_learned_thr060_adaptive416_512_switch075"
$SummaryDir = "runs\edge_cloud\summary_${RunTag}"
$CostDir = "runs\edge_cloud\cost_${RunTag}"

Write-Host "RepoRoot   : $RepoRoot"
Write-Host "DataRoot   : $DataRoot"
Write-Host "TrainList  : $TrainList"
Write-Host "ValList    : $ValList"
Write-Host "PolicyDir  : $PolicyDir"
Write-Host "Device     : $Device"
Write-Host ""
Write-Host "Expected time on a single GPU:" -ForegroundColor Yellow
Write-Host "- full train2017 learned uploader labeling + fit: about 7 to 10 hours"
Write-Host "- each val2017 evaluation run: about 15 to 30 minutes"
Write-Host "- full pipeline with 3 learned runs + 1 rule run: about 9 to 12 hours"

if (-not $SkipTrain) {
    Invoke-Step -Title "Train learned uploader on COCO train2017" -Args @(
        "examples\YOLO26-Edge-Cloud\train_learned_uploader.py",
        "--source", $TrainList,
        "--edge-model", "yolo26s.pt",
        "--cloud-model", "yolo26x.pt",
        "--device", $Device,
        "--enable-adaptive-roi",
        "--roi-small-gain", "1.6",
        "--roi-edge-gain", "1.45",
        "--roi-large-gain", "1.15",
        "--roi-extra-cloud-conf", "0.55",
        "--roi-veto-edge-conf", "0.25",
        "--max-roi-candidates", "2",
        "--save-dir", $PolicyDir
    )
}

if (-not (Test-Path $PolicyPath)) {
    throw "Learned policy not found: $PolicyPath"
}

if (-not $SkipRuleBaseline) {
    Invoke-Step -Title "Evaluate rule aggressive baseline on COCO val2017" -Args @(
        "examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py",
        "--source", $ValList,
        "--edge-model", "yolo26s.pt",
        "--cloud-model", "yolo26x.pt",
        "--device", $Device,
        "--conf", "0.25",
        "--cloud-conf", "0.25",
        "--upload-threshold", "0.35",
        "--max-roi-per-image", "1",
        "--keyframe-interval", "30",
        "--enable-adaptive-roi",
        "--roi-small-gain", "1.6",
        "--roi-edge-gain", "1.45",
        "--roi-large-gain", "1.15",
        "--roi-extra-cloud-conf", "0.55",
        "--roi-veto-edge-conf", "0.25",
        "--enable-complexity-probe",
        "--complexity-threshold", "0.10",
        "--empty-complexity-threshold", "0.05",
        "--complexity-max-detections", "3",
        "--complexity-coverage-threshold", "0.06",
        "--save-dir", $RuleDir
    )
}

Invoke-Step -Title "Evaluate learned uploader thr0.70 cloud512 on COCO val2017" -Args @(
    "examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py",
    "--source", $ValList,
    "--edge-model", "yolo26s.pt",
    "--cloud-model", "yolo26x.pt",
    "--device", $Device,
    "--conf", "0.25",
    "--cloud-conf", "0.25",
    "--cloud-imgsz", "512",
    "--enable-learned-policy",
    "--learned-policy-model", $PolicyPath,
    "--learned-frame-threshold", "0.70",
    "--learned-roi-threshold", "0.65",
    "--learned-cost-weight", "0.02",
    "--learned-max-roi-per-image", "1",
    "--enable-adaptive-roi",
    "--roi-small-gain", "1.6",
    "--roi-edge-gain", "1.45",
    "--roi-large-gain", "1.15",
    "--roi-extra-cloud-conf", "0.55",
    "--roi-veto-edge-conf", "0.25",
    "--save-dir", $Learned70Dir
)

Invoke-Step -Title "Evaluate learned uploader thr0.60 cloud512 on COCO val2017" -Args @(
    "examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py",
    "--source", $ValList,
    "--edge-model", "yolo26s.pt",
    "--cloud-model", "yolo26x.pt",
    "--device", $Device,
    "--conf", "0.25",
    "--cloud-conf", "0.25",
    "--cloud-imgsz", "512",
    "--enable-learned-policy",
    "--learned-policy-model", $PolicyPath,
    "--learned-frame-threshold", "0.60",
    "--learned-roi-threshold", "0.65",
    "--learned-cost-weight", "0.02",
    "--learned-max-roi-per-image", "1",
    "--enable-adaptive-roi",
    "--roi-small-gain", "1.6",
    "--roi-edge-gain", "1.45",
    "--roi-large-gain", "1.15",
    "--roi-extra-cloud-conf", "0.55",
    "--roi-veto-edge-conf", "0.25",
    "--save-dir", $Learned60Dir
)

Invoke-Step -Title "Evaluate learned uploader thr0.60 adaptive 416/512 on COCO val2017" -Args @(
    "examples\YOLO26-Edge-Cloud\evaluate_edge_cloud.py",
    "--source", $ValList,
    "--edge-model", "yolo26s.pt",
    "--cloud-model", "yolo26x.pt",
    "--device", $Device,
    "--conf", "0.25",
    "--cloud-conf", "0.25",
    "--cloud-imgsz", "512",
    "--enable-learned-policy",
    "--learned-policy-model", $PolicyPath,
    "--learned-frame-threshold", "0.60",
    "--learned-roi-threshold", "0.65",
    "--learned-cost-weight", "0.02",
    "--learned-max-roi-per-image", "1",
    "--enable-learned-adaptive-cloud-imgsz",
    "--learned-cloud-imgsz-low", "416",
    "--learned-cloud-imgsz-high", "512",
    "--learned-cloud-imgsz-switch", "0.75",
    "--enable-adaptive-roi",
    "--roi-small-gain", "1.6",
    "--roi-edge-gain", "1.45",
    "--roi-large-gain", "1.15",
    "--roi-extra-cloud-conf", "0.55",
    "--roi-veto-edge-conf", "0.25",
    "--save-dir", $AdaptiveDir
)

if (-not $SkipSummary) {
    $SummaryRuns = @()
    if (-not $SkipRuleBaseline) {
        $SummaryRuns += $RuleDir
    }
    $SummaryRuns += @($Learned70Dir, $Learned60Dir, $AdaptiveDir)

    Invoke-Step -Title "Summarize COCO full runs" -Args @(
        "examples\YOLO26-Edge-Cloud\summarize_strategy_results.py"
    ) + $SummaryRuns + @(
        "--save-dir", $SummaryDir
    )

    Invoke-Step -Title "Analyze COCO full run costs" -Args @(
        "examples\YOLO26-Edge-Cloud\analyze_edge_cloud_costs.py",
        "--summary-csv", (Join-Path $SummaryDir "strategy_summary.csv"),
        "--save-dir", $CostDir,
        "--edge-model", "yolo26s.pt",
        "--cloud-model", "yolo26x.pt",
        "--learned-policy", $PolicyPath,
        "--imgsz", "640"
    )
}

Write-Host ""
Write-Host "Pipeline completed." -ForegroundColor Green
Write-Host "Policy : $PolicyPath"
Write-Host "Summary: $(Join-Path $SummaryDir 'strategy_summary.csv')"
Write-Host "Cost   : $(Join-Path $CostDir 'cost_analysis.csv')"
