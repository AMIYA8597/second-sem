$ErrorActionPreference = 'Continue'

$root = 'D:\work\second-sem\Software-Testing'
$outDir = Join-Path $root '_generated\source_texts'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$manifest = @()

function Save-TextFile {
    param(
        [string]$Path,
        [string]$Text
    )
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Set-Content -Path $Path -Value $Text -Encoding UTF8
}

# Extract from PPT/PPTX
$pp = $null
try {
    $pp = New-Object -ComObject PowerPoint.Application
} catch {
    Write-Output "PowerPoint COM unavailable: $($_.Exception.Message)"
}

if ($pp -ne $null) {
    try {
        $pp.DisplayAlerts = 1
    } catch {
    }
    Get-ChildItem -Path $root -Recurse -File | Where-Object { $_.Extension -in '.ppt', '.pptx' } | ForEach-Object {
        $file = $_
        $status = 'ok'
        $text = ''
        try {
            $pres = $pp.Presentations.Open($file.FullName, $false, $true, $false)
            $slideNum = 0
            foreach ($slide in $pres.Slides) {
                $slideNum++
                $chunks = @()
                foreach ($shape in $slide.Shapes) {
                    try {
                        if ($shape.HasTextFrame -eq -1) {
                            if ($shape.TextFrame.HasText -eq -1) {
                                $chunks += $shape.TextFrame.TextRange.Text
                            }
                        }
                    } catch {
                    }
                }
                $slideText = ($chunks -join " `n").Trim()
                $text += "`n=== SLIDE $slideNum ===`n$slideText`n"
            }
            $pres.Close()
        } catch {
            $status = "error: $($_.Exception.Message)"
        }

        $safeName = ($file.FullName.Substring($root.Length).TrimStart('\\') -replace '[\\/:*?""<>|]', '_') + '.txt'
        $outPath = Join-Path $outDir $safeName
        Save-TextFile -Path $outPath -Text $text

        $manifest += [PSCustomObject]@{
            Source = $file.FullName
            Type = 'slides'
            Output = $outPath
            Status = $status
        }
    }

    $pp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
}

# Extract from PDF via Word conversion/read
$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
} catch {
    Write-Output "Word COM unavailable: $($_.Exception.Message)"
}

if ($word -ne $null) {
    try {
        $word.DisplayAlerts = 0
    } catch {
    }
    Get-ChildItem -Path $root -Recurse -File -Filter *.pdf | ForEach-Object {
        $file = $_
        $status = 'ok'
        $text = ''
        try {
            $doc = $word.Documents.Open($file.FullName, $false, $true, $false, '', '', $true, '', '', 0, '', $false, $false, 0, $true, $false)
            $text = $doc.Content.Text
            $doc.Close($false)
        } catch {
            $status = "error: $($_.Exception.Message)"
            try {
                if ($doc -ne $null) { $doc.Close($false) }
            } catch {
            }
        }

        $safeName = ($file.FullName.Substring($root.Length).TrimStart('\\') -replace '[\\/:*?""<>|]', '_') + '.txt'
        $outPath = Join-Path $outDir $safeName
        Save-TextFile -Path $outPath -Text $text

        $manifest += [PSCustomObject]@{
            Source = $file.FullName
            Type = 'pdf'
            Output = $outPath
            Status = $status
        }
    }

    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}

$manifestPath = Join-Path $root '_generated\extraction_manifest.csv'
$manifest | Export-Csv -Path $manifestPath -NoTypeInformation -Encoding UTF8
Write-Output "WROTE_MANIFEST: $manifestPath"
