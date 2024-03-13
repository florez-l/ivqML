// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Trainer__CommandLine__h__
#define __ivqML__Trainer__CommandLine__h__

namespace ivqML
{
  namespace Trainer
  {
    /**
     */
    template< class _TOptimizer >
    class CommandLine
      : public _TOptimizer
    {
    public:
      using Self       = CommandLine;
      using Superclass = _TOptimizer;
      using TCost      = typename Superclass::TCost;
      using TModel     = typename Superclass::TModel;
      using TScalar    = typename Superclass::TScalar;
      using TNatural   = typename Superclass::TNatural;
      using TMat       = typename Superclass::TMat;
      using TCol       = typename Superclass::TCol;
      using TRow       = typename Superclass::TRow;
      using TMatMap    = typename Superclass::TMatMap;
      using TColMap    = typename Superclass::TColMap;
      using TRowMap    = typename Superclass::TRowMap;
      using TMatCMap   = typename Superclass::TMatCMap;
      using TColCMap   = typename Superclass::TColCMap;
      using TRowCMap   = typename Superclass::TRowCMap;

    public:
      CommandLine( );
      virtual ~CommandLine( ) = default;

      static bool debugger(
        const TScalar& J,
        const TScalar& G,
        const TModel* m,
        const TNatural& i,
        bool d
        );

    protected:
      static bool s_Stop;
    };
  } // end namespace
} // end namespace

#include <ivqML/Trainer/CommandLine.hxx>

#endif // __ivqML__Trainer__CommandLine__h__

// eof - $RCSfile$
