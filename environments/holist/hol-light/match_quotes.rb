def next_nonspace(s, i)
  while i < s.size and s[i] =~ /\s/ do
    i += 1
  end
  return i
end
    
def find_closing_quote(s, i)
  start = i
  escaped = false
  while i < s.size do
    if s[i] == "\\" then
      escaped = !escaped
    else
      if s[i] == "\"" and !escaped then
        return i
      end
      escaped = false
    end
    i += 1
  end
  raise "Unterminated quote."
end

ARGV.each do |fname|
  s = IO.read(fname)
  system("cp #{fname} #{fname}.bak6")
  last = 0
  File.open(fname + ".out", "w") do |f|
    s.to_enum(:scan,/[^"]CONV_TAC|GEN_REWRITE_TAC/).map do |m,|
      match = $`
      index = match.size
      start = index + m.size
      p = next_nonspace(s, start)
      c = start
      if s[p] != "\"" then
        start += 1
        c = start
      else
        c = find_closing_quote(s, p+1) + 1
      end
      ss = s[last...start]
      f.print ss
      last = c
    end
    f.puts s[last..-1]
  end
end
