ARGV.each do |fn|
  s = IO.read(fn)
  if s =~ /CONV_TAC|GEN_REWRITE_TAC/ then
    puts "git add #{fn}"
  end
end
