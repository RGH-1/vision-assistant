param(
    [string]$FixtureDir = "fixtures/recorded_session"
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$fixtureRoot = Join-Path $projectRoot $FixtureDir
$dumpPath = Join-Path $fixtureRoot "pg_dump"
$imagesPath = Join-Path $fixtureRoot "images"
$envPath = Join-Path $projectRoot ".env"

if (!(Test-Path -LiteralPath $dumpPath) -or !(Test-Path -LiteralPath $imagesPath)) {
    throw "Fixture is incomplete. Expected pg_dump and images under $fixtureRoot"
}
if (!(Test-Path -LiteralPath $envPath)) {
    throw "Create .env from .env.example before restoring a fixture."
}

$settings = @{}
Get-Content -LiteralPath $envPath | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') { $settings[$matches[1].Trim()] = $matches[2].Trim() }
}
foreach ($name in "POSTGRES_USER", "POSTGRES_DB", "DATA_DIR") {
    if (!$settings[$name]) { throw "$name is missing from .env" }
}

$composeFile = Join-Path $projectRoot "infra/docker-compose.yml"
docker compose --env-file $envPath -f $composeFile up -d db
$container = (docker compose --env-file $envPath -f $composeFile ps -q db).Trim()
if (!$container) { throw "PostgreSQL container did not start." }

docker cp $dumpPath "${container}:/tmp/vision-assistant-fixture.dump"
docker exec $container pg_restore -U $settings["POSTGRES_USER"] -d $settings["POSTGRES_DB"] --clean --if-exists --no-owner /tmp/vision-assistant-fixture.dump
docker exec $container rm /tmp/vision-assistant-fixture.dump

$dataRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot $settings["DATA_DIR"]))
foreach ($name in "frames", "thumbs", "originals") {
    $target = [IO.Path]::GetFullPath((Join-Path $dataRoot $name))
    if (!$target.StartsWith($dataRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside DATA_DIR: $target"
    }
    Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
}
Copy-Item -Path (Join-Path $imagesPath "*") -Destination $dataRoot -Recurse -Force
Write-Host "Restored fixture from $fixtureRoot"
