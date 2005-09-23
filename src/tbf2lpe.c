#define NAME "tbf2lpe"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <aterm2.h>
#include "gslowlevel.h"
#include "gsfunc.h"
#include "lpetrans.h"

static void print_help(FILE *f)
{
  fprintf(f,
    "Usage: %s OPTIONS [INFILE [OUTFILE]]\n", NAME);
  fprintf(f,
    "Read mCRL LPE from INFILE, convert it to a mCRL2 LPE and save the result to\n"
    "OUTFILE. If OUTFILE is not present, stdout is used. If INFILE is not present,\n"
    "stdin is used. To use stdin and save the output to a file, use '-' for INFILE.\n"
    "\n"
    "The OPTIONS that can be used are:\n"
    "-h, --help               display this help message\n"
    "-q, --quiet              do not print any unrequested information\n"
    "-v, --verbose            display extra information about the conversion process\n"
      );
}

int main(int argc, char **argv)
{
	FILE *InStream, *OutStream;
	ATerm bot;
	#define sopts "hqv"
	struct option lopts[] = {
		{ "help",		no_argument,	NULL,	'h' },
		{ "quiet",		no_argument,	NULL,	'q' },
		{ "verbose",		no_argument,	NULL,	'v' },
 		{ 0, 0, 0, 0 }
	};
	int opt;
	bool opt_quiet,opt_verbose;
	ATerm mu_spec,spec;

	ATinit(argc,argv,&bot);
	gsEnableConstructorFunctions();

	opt_quiet = false;
	opt_verbose = false;
	while ( (opt = getopt_long(argc,argv,sopts,lopts,NULL)) != -1 )
	{
		switch ( opt )
		{
			case 'h':
				print_help(stderr);
				return 0;
			case 'q':
				opt_quiet = true;
				break;
			case 'v':
				opt_verbose = true;
				break;
			default:
				break;
		}
	}
	if ( opt_quiet && opt_verbose )
	{
		gsErrorMsg("options -q/--quiet and -v/--verbose cannot be used together\n");
		return 1;
	}
	if ( opt_quiet )
		gsSetQuietMsg();
	if ( opt_verbose )
		gsSetVerboseMsg();

	InStream = stdin;
	if ( optind < argc && strcmp(argv[optind],"-") )
	{
		if ( (InStream = fopen(argv[optind],"r")) == NULL )
		{
			gsErrorMsg("cannot open file '%s' for reading\n",argv[optind]);
			return 1;
		}
	}

	OutStream = stdout;
	if ( optind+1 < argc )
	{
		if ( (OutStream = fopen(argv[optind+1],"w")) == NULL )
		{
			gsErrorMsg("cannot open file '%s' for writing\n",argv[optind+1]);
			return 1;
		}
	}

	if ( InStream == stdin )
		gsVerboseMsg("reading mCRL LPE from stdin...\n");
	else
		gsVerboseMsg("reading mCRL LPE from '%s'...\n",argv[optind]);
	mu_spec = ATreadFromFile(InStream);
	if ( mu_spec == NULL )
	{
		gsErrorMsg("input is not a valid mCRL LPE file\n");
		return 1;
	}

	spec = (ATerm) translate((ATermAppl) mu_spec);

	if ( OutStream == stdout )
		gsVerboseMsg("writing mCRL2 LPE to stdout...\n");
	else
		gsVerboseMsg("writing mCRL2 LPE to '%s'...\n",argv[optind+1]);
	ATwriteToBinaryFile(spec,OutStream);

  	return 0;
}
