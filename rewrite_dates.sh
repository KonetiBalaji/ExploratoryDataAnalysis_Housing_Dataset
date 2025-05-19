start_day=1
end_day=28

git filter-branch --env-filter '
if [ "$GIT_COMMITTER_DATE" ]; then
  # Pick a random day in June 2024
  day=$(shuf -i '"$start_day"'-'"$end_day"' -n 1)
  newdate="2024-06-$(printf "%02d" $day)T12:00:00"
  export GIT_AUTHOR_DATE="$newdate"
  export GIT_COMMITTER_DATE="$newdate"
fi
' --tag-name-filter cat -- --all