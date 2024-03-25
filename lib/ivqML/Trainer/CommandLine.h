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
      using TScl       = typename Superclass::TScl;
      using TNat       = typename Superclass::TNat;
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
      ivqMLAttributeMacro( debug_step, TNat, 100 );

    public:
      CommandLine( );
      virtual ~CommandLine( ) = default;

      virtual void register_options(
        boost::program_options::options_description& opt
        ) override;

      static bool debugger(
        const TModel* model,
        const TScl& norm, const TNat& iter,
        const TCost* cost,
        bool force
        );

    protected:
      static bool s_Stop;
      static TNat m_DebugStep;
    };
  } // end namespace
} // end namespace

#include <ivqML/Trainer/CommandLine.hxx>

#endif // __ivqML__Trainer__CommandLine__h__

// eof - $RCSfile$
