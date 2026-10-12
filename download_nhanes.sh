#!/usr/bin/env bash
# 下载 NHANES 常规体检/血液检验数据（论文1复现用）
# 数据源：美国 CDC NHANES 公开数据，无需注册、无需伦理审批
# 用法：bash download_nhanes.sh
set -uo pipefail

BASE="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
OUTROOT="$(cd "$(dirname "$0")" && pwd)/raw"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"

# 周期:年份 后缀
CYCLES=("2013-2014:H" "2015-2016:I" "2017-2018:J" "2021-2023:L")

# 文件清单：文件名(不含后缀) 用途
FILES=(
  "DEMO:人口学 年龄性别种族收入"
  "CBC:血常规 白细胞分类 红细胞 血红蛋白 血小板 MCV MCH MCHC"
  "BIOPRO:生化全套 肝肾功能 电解质 白蛋白 尿酸"
  "GHB:糖化血红蛋白"
  "GLU:空腹血糖 空腹子样本"
  "INS:空腹胰岛素 空腹子样本"
  "HDL:高密度脂蛋白"
  "TRIGLY:甘油三酯"
  "TCHOL:总胆固醇"
  "HSCRP:高敏C反应蛋白"
  "BMX:体格测量 BMI 腰围"
  "BPQ:血压问卷"
  "DIQ:糖尿病问卷"
  "MCQ:慢性病问卷 心血管 癌症 等"
  "KIQ_U:肾脏病问卷"
  "WHQ:体重史"
  "ALQ:饮酒"
  "SMQ:吸烟"
)

mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/../manifest.tsv"
: > "$MANIFEST"
printf "cycle\tfile\tbytes\tstatus\n" >> "$MANIFEST"

ok=0; miss=0
for entry in "${CYCLES[@]}"; do
  cyc="${entry%%:*}"; suf="${entry##*:}"
  yyyy="${cyc%%-*}"
  dir="$OUTROOT/$cyc"
  mkdir -p "$dir"
  echo "=================== $cyc (后缀 $suf) ==================="
  for fe in "${FILES[@]}"; do
    fn="${fe%%:*}"; desc="${fe##*:}"
    url="$BASE/$yyyy/DataFiles/${fn}_${suf}.xpt"
    dest="$dir/${fn}_${suf}.xpt"
    if [ -s "$dest" ]; then
      sz=$(stat -f%z "$dest" 2>/dev/null || echo 0)
      printf "  %-16s %9s B  (已存在)\n" "${fn}_${suf}.xpt" "$sz"
      printf "%s\t%s\t%s\t%s\n" "$cyc" "${fn}_${suf}.xpt" "$sz" "cached" >> "$MANIFEST"
      ok=$((ok+1)); continue
    fi
    code=$(curl -sL -o "$dest" -w "%{http_code}" --max-time 60 -A "$UA" "$url" 2>/dev/null)
    if [ "$code" = "200" ] && [ -s "$dest" ]; then
      sz=$(stat -f%z "$dest")
      # 校验不是 HTML 错误页
      if head -c 15 "$dest" | grep -qi "<!DOCTYPE\|<html"; then
        rm -f "$dest"; printf "  %-16s %s\n" "${fn}_${suf}.xpt" "HTML错误页 跳过"; miss=$((miss+1))
        printf "%s\t%s\t%s\t%s\n" "$cyc" "${fn}_${suf}.xpt" "0" "html-error" >> "$MANIFEST"
      else
        printf "  %-16s %9s B  %s\n" "${fn}_${suf}.xpt" "$sz" "$desc"
        printf "%s\t%s\t%s\t%s\n" "$cyc" "${fn}_${suf}.xpt" "$sz" "ok" >> "$MANIFEST"
        ok=$((ok+1))
      fi
    else
      rm -f "$dest"
      printf "  %-16s %s\n" "${fn}_${suf}.xpt" "不存在 (HTTP $code)"
      printf "%s\t%s\t%s\t%s\n" "$cyc" "${fn}_${suf}.xpt" "0" "missing" >> "$MANIFEST"
      miss=$((miss+1))
    fi
  done
done

echo
echo "完成：下载 $ok 个，缺失 $miss 个"
echo "清单：$MANIFEST"
du -sh "$OUTROOT" 2>/dev/null
