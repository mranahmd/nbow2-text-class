if($#ARGV != 2){
	print STDERR "\nUSAGE: $_0 <task-vocab> <glove-vocab> <glove-word-vectors>\n";
	exit(0);
}

binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");


my @zeroes = (0.0) x 300; 

my %newVocab;
my $i=0;
open FILE, '<:encoding(UTF-8)', $ARGV[0] or die "Can't open $ARGV[0] for reading: $!";
while(<FILE>){
	my $line = $_;
	$line =~ s/\r*\n+//;
	$newVocab{$line} = $i;
	$i++;
}
close FILE;

%vectors;

open FILEV, '<:encoding(UTF-8)', $ARGV[1] or die "Can't open $ARGV[1] for reading: $!";
open FILEM, '<:encoding(UTF-8)', $ARGV[2] or die "Can't open $ARGV[2] for reading: $!";
while(!eof(FILEV) && !eof(FILEM)){
	my $w = <FILEV>;
	$w =~ s/\r*\n+//;
	my $vec = <FILEM>;
	if(exists($newVocab{$w})){
		$vectors{$w} = $vec;
	}
}
close FILEV;
close FILEM;

open FILE, '<:encoding(UTF-8)', $ARGV[0] or die "Can't open $ARGV[0] for reading: $!";
while(<FILE>){
	my $line = $_;
	$line =~ s/\r*\n+//;
	if(exists($vectors{$line})){
		print $vectors{$line};
	}else{
		print join(" ", @zeroes)."\n"; 
		print STDERR $newVocab{$line},"\n";
	}
}
close FILE;

