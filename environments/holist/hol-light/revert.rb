ARGV.each do |fn|
  puts "cp #{fn} #{fn.gsub(/\.bak5/,'')}"
end
