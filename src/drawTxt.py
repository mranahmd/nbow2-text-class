import codecs

def printDoc(fname, lineNum):
	colorScale = ["#ffff00", 	#1.0
				"#ffcc66",
				"#ff6600",
				"#ff9966",
				"#ff0000",		#0.5
				"#ff6666",
				"#cc0099",
				"#9900cc",
				"#6600cc",
				"#333399"]		#0.0
	l = 0
	out = "<font size=\"5\" color=\"red\"> Not so many lines in file! </font>"
	f = codecs.open(fname, "r", "utf-8")
	for line in iter(f):
		l = l+1
		if l == lineNum:
			out = ""
			for ww in line.split():
				wrd, wt = ww.split('_')
				wsize = int(float(wt) * 10) + 1
				wcolor = 10 - wsize
				wsize = wsize - 3
				if wsize < 1:
					wsize = 1
				currWrd = "<font size=\"" + str(wsize) +"\" color=\"" + colorScale[wcolor] + "\">" + wrd + "</font>"
				out = out + " " + currWrd
			break	
	f.close()
	return out + " <br> "

