print "b"
import cgitb, cgi, plot
cgitb.enable()
print "a"
form = cgi.FieldStorage()

m = form['m']
n = form['n']

plot.graphpict(m, n)