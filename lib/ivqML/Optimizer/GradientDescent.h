// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__h__
#define __ivqML__Optimizer__GradientDescent__h__

#include <ivqML/Optimizer/Base.h>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TCost >
    class GradientDescent
      : public ivqML::Optimizer::Base< _TCost >
    {
    public:
      using TCost      = _TCost;
      using Self       = GradientDescent;
      using Superclass = ivqML::Optimizer::Base< TCost >;
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
      ivqMLAttributeMacro( alpha, TScl, 1e-4 );

    public:
      GradientDescent( );
      virtual ~GradientDescent( );

      virtual void register_options(
        boost::program_options::options_description& opt
        ) override;

      virtual void fit( TModel& model ) override;

    private:
      GradientDescent( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;
    };
  } // end namespace
} // end namespace

#include <ivqML/Optimizer/GradientDescent.hxx>

#endif // __ivqML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
