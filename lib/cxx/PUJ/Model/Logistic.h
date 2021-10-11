// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Logistic__h__
#define __PUJ__Logistic__h__

#include <vector>
#include <PUJ/Model/Linear.h>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
    class Logistic
      : public Linear< _TScalar, _TTraits >
    {
    public:
      PUJ_TraitsMacro( Logistic );
      using Superclass = PUJ::Model::Linear< _TScalar, _TTraits >;

    public:
      Logistic( const TRow& w, const TScalar& b );
      virtual ~Logistic( ) = default;

      virtual TScalar operator()( const TRow& x ) const override;
      virtual TCol operator()( const TMatrix& x ) const override;

    public:
      /**
       */
      class Cost
      {
      public:
        Cost( const TMatrix& X, const TCol& y );
        virtual ~Cost( ) = default;

        TScalar operator()( const TRow& t, TRow* g = nullptr ) const;

      protected:
        TMatrix m_X;
        TCol    m_y;
        TRow    m_Xy;
        TScalar m_uy;
        std::vector< unsigned long long > m_Zeros;
        std::vector< unsigned long long > m_Ones;
      };
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Logistic__h__

// eof - $RCSfile$
