// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Linear__h__
#define __PUJ_ML__Model__Linear__h__

#include <PUJ_ML/Model/Base.h>

namespace PUJ_ML
{
  namespace Model
  {
    /**
     */
    template< class _R >
    class Linear
      : public Model::Base< _R >
    {
    public:
      using Self = Linear;
      using Superclass = Model::Base< _R >;
      using TReal = typename Superclass::TReal;
      using TMatrix = typename Superclass::TMatrix;
      using TCol = typename Superclass::TCol;
      using TRow = typename Superclass::TRow;

    public:
      Linear( const unsigned long long& n = 1 );
      virtual ~Linear( );

      virtual void set_number_of_parameters(
        const unsigned long long& n
        ) override;

      template< class _Y, class _X >
      void evaluate(
        Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
        ) const;

    protected:
      Eigen::Map< TCol >* m_T { nullptr };
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/Linear.hxx>

#endif // __PUJ_ML__Model__Linear__h__

// eof - $RCSfile$
