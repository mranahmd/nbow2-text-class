if($#ARGV !=1 ){
	print STDERR "\nUSAGE: $0 <corpus-file> <vocab-file> \n\n";
	exit(0);
}

$index=-1;
%vocabIndex;
open FILE, $ARGV[1] or die $!;
while (<FILE>) {
	$index++;
	$line =  $_;
	$line =~ s/^\s+//g;
	$line =~ s/\r*\n*$//;
	$vocabIndex{$line} = $index;
}
close(FILE);

open FILE, $ARGV[0] or die $!;
while (<FILE>) {
	my $out = "0 ";
	my @words = split('\s+', $_);
	foreach $word (@words){
		if(exists $vocabIndex{$word}) {
				$out = $out." ".$vocabIndex{$word};
		}
	}
	print STDOUT "$out 0\n";
}
close(FILE);
