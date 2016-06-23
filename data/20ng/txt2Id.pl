if($#ARGV != 2){
	print STDERR "\nUSAGE: $_0 <worddict> <labeldict> <20ng-txt>\n";
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

my %labeldict;
$i=0;
open FILE, '<:encoding(UTF-8)', $ARGV[1] or die "Can't open $ARGV[1] for reading: $!";
while(<FILE>){
	my $line = $_;
	$line =~ s/\r*\n+//;
	$labeldict{$line} = $i;
	$i++;
}
close FILE;

open FILE, '<:encoding(UTF-8)', $ARGV[2] or die "Can't open $ARGV[2] for reading: $!";
while(<FILE>){
	my $line = $_;
	$line =~ s/\r*\n+//;
	my ($lab, $txt) = split(/\t/, $line);
	print STDOUT $labeldict{$lab}."\n";
	my @tmp = split(/\s+/, $txt);
	foreach my $w (@tmp){
		if(exists($worddict{$w})){
			print STDERR $worddict{$w}." ";
		}
	}
	print STDERR "\n";
}
close FILE;
