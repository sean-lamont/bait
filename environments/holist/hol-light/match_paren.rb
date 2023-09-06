def next_nonspace(s, i)
  while i < s.size and s[i] =~ /\s/ do
    i += 1
  end
  return i
end
    
def find_closing_paren (s, i)
  count = 0
  while i < s.size 
    if s[i] == "("
      count += 1
    elsif s[i] == ")"
      count -= 1
    end
    if count <= 0 then
      return i
    end
    i += 1
  end
end
    
def match_nonalnum(s, i)
  while i < s.size 
    if s[i] =~ /\w/ then
      i += 1
    else
      return i
    end
  end
end
    
counts = {}

def sanitize(s)
  return s.gsub(/\s+/, " ").gsub(/"/, "\\\"")
end

ARGV.each do |fname|
  s = IO.read(fname)
  # system("cp #{fname} #{fname}.bak7")
  last = 0
  File.open(fname + ".out", "w") do |f|
    s.to_enum(:scan,/[^"](CONV_TAC|GEN_REWRITE_TAC)/).map do |m,|    
      index = $`.size
      token = $1
      start = index + m.size
      p = next_nonspace(s, start)
      c = start
      param = ""
      if s[p] == "(" then
        c = find_closing_paren(s, p)
        param = "(" + s[p+1...c] + ")"
        c += 1
      else
        c = match_nonalnum(s, p)
        param = s[p...c]
      end
      f.print "#{s[last...index]}#{token} \"#{sanitize(param)}\" #{param}"
      last = c
    end
    f.puts s[last..-1]
  end
end
