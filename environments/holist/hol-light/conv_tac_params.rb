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

ARGV.each do |fname|
  s = IO.read(fname)
  s.to_enum(:scan,/CONV_TAC/).map do |m,|
    index = $`.size
    start = index + m.size
    p = next_nonspace(s, start)
    # puts "#{index} #{start} #{s[p]} #{p}"
    c = start
    if s[p] == "(" then
      # puts "A"
      c = find_closing_paren(s, p)
      puts(s[p+1...c].gsub(/\s+/, " ") + " " + fname)
    elsif s[0] =~ /\w/ then
      c = match_nonalnum(s, p)
      puts(s[p...c] + " " + fname)
    end
  end
end
