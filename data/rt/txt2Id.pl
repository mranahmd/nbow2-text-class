if($#ARGV != 1){
	print STDERR "\nUSAGE: $_0 <vocab> <txt>\n";
	exit(0);
}

my %worddict;
my $i=0;
open FILE, '<:encoding(UTF-8)', $ARGV[0] or die "Can't open $ARGV[0] for reading: $!";
while(<FILE>){
	my $line = $_;
	$line =~ s/\r*\n+//;
	$worddict{$line} = $i;
	$i++;
}
close FILE;

open FILE, '<:encoding(UTF-8)', $ARGV[1] or die "Can't open $ARGV[1] for reading: $!";
while(<FILE>){
	my $line = $_;
	$line =~ s/\r*\n+//;
	my @tmp = split(/\s+/, $line);
	foreach my $w (@tmp){
		if(exists($worddict{$w})){
			print $worddict{$w}." ";
		}
	}
	print "\n";
}
close FILE;
